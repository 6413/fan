#include <pch.h>
#include "codegen.h"

#include "../include/KaleidoscopeJIT.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include "parser.h"

using namespace llvm;
using namespace llvm::orc;
using namespace ast;

ExitOnError ExitOnErr;
std::map<std::string, AllocaInst*> NamedValues;
std::unique_ptr<KaleidoscopeJIT> TheJIT;
std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

std::unique_ptr<LLVMContext> TheContext;
std::unique_ptr<Module> TheModule;
std::unique_ptr<IRBuilder<>> Builder;

std::unique_ptr<DIBuilder> DBuilder;

DebugInfo KSDbgInfo;


//===----------------------------------------------------------------------===//
// Debug Info Support
//===----------------------------------------------------------------------===//

DIType* DebugInfo::getDoubleTy() {
  if (DblTy)
    return DblTy;

  DblTy = DBuilder->createBasicType("double", 64, dwarf::DW_ATE_float);
  return DblTy;
}

static DISubroutineType* CreateFunctionType(unsigned NumArgs) {
  SmallVector<Metadata*, 8> EltTys;
  DIType* DblTy = KSDbgInfo.getDoubleTy();

  // Add the result type.
  EltTys.push_back(DblTy);

  for (unsigned i = 0, e = NumArgs; i != e; ++i)
    EltTys.push_back(DblTy);

  return DBuilder->createSubroutineType(DBuilder->getOrCreateTypeArray(EltTys));
}


//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

void codegen_init() {

  Builder = std::unique_ptr<IRBuilder<>>{};

  ExitOnErr = ExitOnError{};
  NamedValues = std::map<std::string, AllocaInst*>{};
  TheJIT = std::unique_ptr<KaleidoscopeJIT>{};
  FunctionProtos = std::map<std::string, std::unique_ptr<PrototypeAST>>{};
}

void create_the_JIT() {
  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
}

std::unique_ptr<KaleidoscopeJIT>& get_JIT() {
  return TheJIT;
}

std::map<std::string, std::unique_ptr<PrototypeAST>>& get_function_protos() {
  return FunctionProtos;
}

void init_module() {
  // Open a new module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(get_JIT()->getDataLayout());

  Builder = std::make_unique<IRBuilder<>>(*TheContext);
}

Value* LogErrorV(const char* Str) {
  LogError(Str);
  return nullptr;
}

Function* getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto* F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return null.
  return nullptr;
}

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static AllocaInst* CreateEntryBlockAlloca(Function* TheFunction,
  StringRef VarName) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
    TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getDoubleTy(*TheContext), nullptr, VarName);
}

Value* NumberExprAST::codegen() {
  KSDbgInfo.emitLocation(this);
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value* VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value* V = NamedValues[Name];
  if (!V)
    return LogErrorV("Unknown variable name");

  KSDbgInfo.emitLocation(this);
  // Load the value.
  // PointerType::getUnqual(i8*)
  return Builder->CreateLoad(Type::getDoubleTy(*TheContext), V, Name.c_str());
}

Value* UnaryExprAST::codegen() {
  Value* OperandV = Operand->codegen();
  if (!OperandV)
    return nullptr;

  Function* F = getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV("Unknown unary operator");

  KSDbgInfo.emitLocation(this);
  return Builder->CreateCall(F, OperandV, "unop");
}

Value* BinaryExprAST::codegen() {
  KSDbgInfo.emitLocation(this);

  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST* LHSE = static_cast<VariableExprAST*>(LHS.get());
    if (!LHSE)
      return LogErrorV("destination of '=' must be a variable");
    // Codegen the RHS.
    Value* Val = RHS->codegen();
    if (!Val)
      return nullptr;

    // Look up the name.
    Value* Variable = NamedValues[LHSE->getName()];
    if (!Variable)
      return LogErrorV("Unknown variable name");

    Builder->CreateStore(Val, Variable);
    return Val;
  }

  Value* L = LHS->codegen();
  Value* R = RHS->codegen();
  if (!L || !R)
    return nullptr;

  switch (Op) {
  case '+':
    return Builder->CreateFAdd(L, R, "addtmp");
  case '-':
    return Builder->CreateFSub(L, R, "subtmp");
  case '*':
    return Builder->CreateFMul(L, R, "multmp");
  case '/': return Builder->CreateFDiv(L, R, "divtmp");
  case '<':
    L = Builder->CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
  case '>':
    L = Builder->CreateFCmpUGT(L, R, "cmptmp");
    return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
  case '%': {
    if (L->getType()->isIntegerTy() && R->getType()->isIntegerTy()) {
      return Builder->CreateSRem(L, R, "modtmp");
    }
    else if (L->getType()->isFloatingPointTy() && R->getType()->isFloatingPointTy()) {
      Function* FloorF = Intrinsic::getDeclaration(TheModule.get(), Intrinsic::floor, Type::getDoubleTy(*TheContext));
      Value* Div = Builder->CreateFDiv(L, R, "divtmp");
      Value* FloorDiv = Builder->CreateCall(FloorF, { Div }, "floordivtmp");
      Value* Mult = Builder->CreateFMul(FloorDiv, R, "multtmp");
      return Builder->CreateFSub(L, Mult, "modtmp");
    }
    else {
      return LogErrorV("Operands to % must be both integers or both floats.");
    }
  }
  default:
    break;
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function* F = getFunction(std::string("binary") + Op);
  assert(F && "binary operator not found!");

  Value* Ops[] = { L, R };
  return Builder->CreateCall(F, Ops, "binop");
}

Value* CallExprAST::codegen() {
  KSDbgInfo.emitLocation(this);

  // Look up the name in the global module table.
  Function* CalleeF = getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value*> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen());
    if (!ArgsV.back())
      return nullptr;
  }

  return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Value* IfExprAST::codegen() {
  KSDbgInfo.emitLocation(this);

  Value* CondV = Cond->codegen();
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  CondV = Builder->CreateFCmpONE(
    CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");

  Function* TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock* ThenBB = BasicBlock::Create(*TheContext, "then", TheFunction);
  BasicBlock* ElseBB = BasicBlock::Create(*TheContext, "else");
  BasicBlock* MergeBB = BasicBlock::Create(*TheContext, "ifcont");

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value.
  Builder->SetInsertPoint(ThenBB);

  Value* ThenV = Then->codegen();
  if (!ThenV)
    return nullptr;

  Builder->CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder->GetInsertBlock();

  // Emit else block.
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);

  Value* ElseV = Else->codegen();
  if (!ElseV)
    return nullptr;

  Builder->CreateBr(MergeBB);
  // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
  ElseBB = Builder->GetInsertBlock();

  // Emit merge block.
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder->SetInsertPoint(MergeBB);
  PHINode* PN = Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, "iftmp");

  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  return PN;
}

// Output for-loop as:
//   var = alloca double
//   ...
//   start = startexpr
//   store start -> var
//   goto loop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   endcond = endexpr
//
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br endcond, loop, endloop
// outloop:
Value* ForExprAST::codegen() {
  Function* TheFunction = Builder->GetInsertBlock()->getParent();
  // Create an alloca for the variable in the entry block.
  AllocaInst* Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
  KSDbgInfo.emitLocation(this);
  // Emit the start code first, without 'variable' in scope.
  Value* StartVal = Start->codegen();
  if (!StartVal)
    return nullptr;
  // Store the value into the alloca.
  Builder->CreateStore(StartVal, Alloca);
  // Make the new basic block for the loop header, inserting after current block.
  BasicBlock* LoopBB = BasicBlock::Create(*TheContext, "loop", TheFunction);
  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(LoopBB);
  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);
  // Within the loop, the variable is defined equal to the PHI node.
  // If it shadows an existing variable, we have to restore it, so save it now.
  AllocaInst* OldVal = NamedValues[VarName];
  NamedValues[VarName] = Alloca;
  // Generate code for the loop body
  if (auto* C = llvm::dyn_cast<CompoundExprAST>(Body.get())) {
    for (auto& Stmt : C->getStatements()) {
      if (!Stmt->codegen())
        return nullptr;
    }
  }
  else {
    if (!Body->codegen())
      return nullptr;
  }

  //// Add printd call inside the loop
  //std::vector<std::unique_ptr<ExprAST>> Args;
  //Args.push_back(std::make_unique<NumberExprAST>(10.0));
  //CallExprAST PrintCall(CurLoc, "printd", std::move(Args));
  //if (!PrintCall.codegen())
  //  return nullptr;

  // Emit the step value.
  Value* StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen();
    if (!StepVal)
      return nullptr;
  }
  else {
    // If not specified, use 1.0.
    StepVal = ConstantFP::get(*TheContext, APFloat(1.0));
  }
  // Compute the end condition.
  Value* EndCond = End->codegen();
  if (!EndCond)
    return nullptr;
  // Reload, increment, and restore the alloca.
  Value* CurVar = Builder->CreateLoad(Type::getDoubleTy(*TheContext), Alloca, VarName.c_str());
  Value* NextVar = Builder->CreateFAdd(CurVar, StepVal, "nextvar");
  Builder->CreateStore(NextVar, Alloca);
  // Convert condition to a bool by comparing non-equal to 0.0.
  EndCond = Builder->CreateFCmpONE(EndCond, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");
  // Create the "after loop" block and insert it.
  BasicBlock* AfterBB = BasicBlock::Create(*TheContext, "afterloop", TheFunction);
  // Insert the conditional branch into the end of LoopEndBB.
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);
  // Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(AfterBB);
  // Restore the unshadowed variable.
  if (OldVal)
    NamedValues[VarName] = OldVal;
  else
    NamedValues.erase(VarName);
  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(*TheContext));
}

Value* VarExprAST::codegen() {
  std::vector<AllocaInst*> OldBindings;

  Function* TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string& VarName = VarNames[i].first;
    ExprAST* Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    //    var a = a in ...   # refers to outer 'a'.
    Value* InitVal;
    if (Init) {
      InitVal = Init->codegen();
      if (!InitVal)
        return nullptr;
    }
    else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }

    AllocaInst* Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
    Builder->CreateStore(InitVal, Alloca);

    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    OldBindings.push_back(NamedValues[VarName]);

    // Remember this binding.
    NamedValues[VarName] = Alloca;
  }

  KSDbgInfo.emitLocation(this);

  // Codegen the body, now that all vars are in scope.
  Value* BodyVal = Body->codegen();
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    NamedValues[VarNames[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
}

Function* PrototypeAST::codegen() {
  // Create a vector to hold the argument types
  std::vector<Type*> ArgTypesLLVM;

  // Iterate through Args and ArgTypes to dynamically determine argument types
  for (size_t i = 0; i < Args.size(); ++i) {
    if (ArgTypes[i] == "double") {
      ArgTypesLLVM.push_back(Type::getDoubleTy(*TheContext));
    }
    else if (ArgTypes[i] == "string") {
      ArgTypesLLVM.push_back(PointerType::getUnqual(Type::getInt8Ty(*TheContext)));
    }
    else {
      LogError("Unknown argument type");
      return nullptr;
    }
  }

  // Create the function type
  FunctionType* FT = FunctionType::get(Type::getDoubleTy(*TheContext), ArgTypesLLVM, false);

  // Create the function
  Function* F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments
  unsigned Idx = 0;
  for (auto& Arg : F->args()) {
    Arg.setName(Args[Idx++]);
  }

  return F;
}

Function* FunctionAST::codegen() {
  auto& P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function* TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // Create a single entry block
  BasicBlock* BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);

  // Debug info setup
  DIFile* Unit = DBuilder->createFile(KSDbgInfo.TheCU->getFilename(),
    KSDbgInfo.TheCU->getDirectory());
  DIScope* FContext = Unit;
  unsigned LineNo = P.getLine();
  unsigned ScopeLine = LineNo;
  DISubprogram* SP = DBuilder->createFunction(FContext, P.getName(), StringRef(), Unit, LineNo, CreateFunctionType(TheFunction->arg_size()), ScopeLine, DINode::FlagPrototyped, DISubprogram::SPFlagDefinition);
  TheFunction->setSubprogram(SP);

  KSDbgInfo.LexicalBlocks.push_back(SP);
  KSDbgInfo.emitLocation(nullptr);

  // Record the function arguments in the NamedValues map
  NamedValues.clear();
  unsigned ArgIdx = 0;
  for (auto& Arg : TheFunction->args()) {
    AllocaInst* Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());
    DILocalVariable* D = DBuilder->createParameterVariable(SP, Arg.getName(), ++ArgIdx, Unit, LineNo, KSDbgInfo.getDoubleTy(), true);
    DBuilder->insertDeclare(Alloca, D, DBuilder->createExpression(), DILocation::get(SP->getContext(), LineNo, 0, SP), Builder->GetInsertBlock());
    Builder->CreateStore(&Arg, Alloca);
    NamedValues[std::string(Arg.getName())] = Alloca;
  }

  KSDbgInfo.emitLocation(Body.get());

  // Generate code for the body
  Value* RetVal = Body->codegen();
  if (!RetVal) {
    TheFunction->eraseFromParent();
    if (P.isBinaryOp())
      BinopPrecedence.erase(Proto->getOperatorName());
    KSDbgInfo.LexicalBlocks.pop_back();
    return nullptr;
  }

  // Create return instruction
  Builder->CreateRet(RetVal);

  KSDbgInfo.LexicalBlocks.pop_back();

  verifyFunction(*TheFunction);

  return TheFunction;
}