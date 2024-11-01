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
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include <cctype>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>
#include <chrono>

using namespace llvm;
using namespace llvm::orc;

//#undef loco_assimp
#include <fan/pch.h>
#include <fan/fmt.h>

#include <chrono>
#include <coroutine>
#include <queue>
#include <iostream>
#include <thread>  // Print thread id


//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,

  // control
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9,
  tok_in = -10,

  // operators
  tok_binary = -11,
  tok_unary = -12,

  // var definition
  tok_var = -13
};

std::string getTokName(int Tok) {
  switch (Tok) {
  case tok_eof:
    return "eof";
  case tok_def:
    return "def";
  case tok_extern:
    return "extern";
  case tok_identifier:
    return "identifier";
  case tok_number:
    return "number";
  case tok_if:
    return "if";
  case tok_then:
    return "then";
  case tok_else:
    return "else";
  case tok_for:
    return "for";
  case tok_in:
    return "in";
  case tok_binary:
    return "binary";
  case tok_unary:
    return "unary";
  case tok_var:
    return "var";
  }
  return std::string(1, (char)Tok);
}

namespace {
  class PrototypeAST;
  class ExprAST;
}

struct DebugInfo {
  DICompileUnit* TheCU;
  DIType* DblTy;
  std::vector<DIScope*> LexicalBlocks;

  void emitLocation(ExprAST* AST);
  DIType* getDoubleTy();
} KSDbgInfo;

struct SourceLocation {
  int Line;
  int Col;
};
static SourceLocation CurLoc;
static SourceLocation LexLoc = { 1, 0 };

std::string code_input = "";

int index = 0;

std::mutex code_input_mutex;

static int advance() {
  int LastChar = code_input.operator[](index);
  //fan::print("processing", LastChar);
  ++index;
  if (index >= code_input.size()) {
    index = 0;
    code_input.clear();
  }

  if (LastChar == '\n' || LastChar == '\r') {
    LexLoc.Line++;
    LexLoc.Col = 0;
  }
  else
    LexLoc.Col++;
  return LastChar;
}

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number

static int gLastChar = ' ';
/// gettok - Return the next token from standard input.
static int gettok() {

  // Skip any whitespace.
  while (isspace(gLastChar))
    gLastChar = advance();

  CurLoc = LexLoc;

  if (isalpha(gLastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = gLastChar;
    while (isalnum((gLastChar = advance())))
      IdentifierStr += gLastChar;

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "extern")
      return tok_extern;
    if (IdentifierStr == "if")
      return tok_if;
    if (IdentifierStr == "then")
      return tok_then;
    if (IdentifierStr == "else")
      return tok_else;
    if (IdentifierStr == "for")
      return tok_for;
    if (IdentifierStr == "in")
      return tok_in;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "var")
      return tok_var;
    return tok_identifier;
  }

  if (isdigit(gLastChar) || gLastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += gLastChar;
      gLastChar = advance();
    } while (isdigit(gLastChar) || gLastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (gLastChar == '#') {
    // Comment until end of line.
    do
      gLastChar = advance();
    while (gLastChar != EOF && gLastChar != '\n' && gLastChar != '\r');

    if (gLastChar != EOF)
      return gettok();
  }

  // Check for end of file.  Don't eat the EOF.
  if (gLastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = gLastChar;
  gLastChar = advance();
  return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//
namespace {

  raw_ostream& indent(raw_ostream& O, int size) {
    return O << std::string(size, ' ');
  }

  /// ExprAST - Base class for all expression nodes.
  class ExprAST {
    SourceLocation Loc;

  public:
    ExprAST(SourceLocation Loc = CurLoc) : Loc(Loc) {}
    virtual ~ExprAST() {}
    virtual Value* codegen() = 0;
    int getLine() const { return Loc.Line; }
    int getCol() const { return Loc.Col; }
    virtual raw_ostream& dump(raw_ostream& out, int ind) {
      return out << ':' << getLine() << ':' << getCol() << '\n';
    }
  };

  /// NumberExprAST - Expression class for numeric literals like "1.0".
  class NumberExprAST : public ExprAST {
    double Val;

  public:
    NumberExprAST(double Val) : Val(Val) {}
    raw_ostream& dump(raw_ostream& out, int ind) override {
      return ExprAST::dump(out << Val, ind);
    }
    Value* codegen() override;
  };

  /// VariableExprAST - Expression class for referencing a variable, like "a".
  class VariableExprAST : public ExprAST {
    std::string Name;

  public:
    VariableExprAST(SourceLocation Loc, const std::string& Name)
      : ExprAST(Loc), Name(Name) {}
    const std::string& getName() const { return Name; }
    Value* codegen() override;
    raw_ostream& dump(raw_ostream& out, int ind) override {
      return ExprAST::dump(out << Name, ind);
    }
  };

  /// UnaryExprAST - Expression class for a unary operator.
  class UnaryExprAST : public ExprAST {
    char Opcode;
    std::unique_ptr<ExprAST> Operand;

  public:
    UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
      : Opcode(Opcode), Operand(std::move(Operand)) {}
    Value* codegen() override;
    raw_ostream& dump(raw_ostream& out, int ind) override {
      ExprAST::dump(out << "unary" << Opcode, ind);
      Operand->dump(out, ind + 1);
      return out;
    }
  };

  /// BinaryExprAST - Expression class for a binary operator.
  class BinaryExprAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;

  public:
    BinaryExprAST(SourceLocation Loc, char Op, std::unique_ptr<ExprAST> LHS,
      std::unique_ptr<ExprAST> RHS)
      : ExprAST(Loc), Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    Value* codegen() override;
    raw_ostream& dump(raw_ostream& out, int ind) override {
      ExprAST::dump(out << "binary" << Op, ind);
      LHS->dump(indent(out, ind) << "LHS:", ind + 1);
      RHS->dump(indent(out, ind) << "RHS:", ind + 1);
      return out;
    }
  };

  /// CallExprAST - Expression class for function calls.
  class CallExprAST : public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;

  public:
    CallExprAST(SourceLocation Loc, const std::string& Callee,
      std::vector<std::unique_ptr<ExprAST>> Args)
      : ExprAST(Loc), Callee(Callee), Args(std::move(Args)) {}
    Value* codegen() override;
    raw_ostream& dump(raw_ostream& out, int ind) override {
      ExprAST::dump(out << "call " << Callee, ind);
      for (const auto& Arg : Args)
        Arg->dump(indent(out, ind + 1), ind + 1);
      return out;
    }
  };

  /// IfExprAST - Expression class for if/then/else.
  class IfExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Cond, Then, Else;

  public:
    IfExprAST(SourceLocation Loc, std::unique_ptr<ExprAST> Cond,
      std::unique_ptr<ExprAST> Then, std::unique_ptr<ExprAST> Else)
      : ExprAST(Loc), Cond(std::move(Cond)), Then(std::move(Then)),
      Else(std::move(Else)) {}
    Value* codegen() override;
    raw_ostream& dump(raw_ostream& out, int ind) override {
      ExprAST::dump(out << "if", ind);
      Cond->dump(indent(out, ind) << "Cond:", ind + 1);
      Then->dump(indent(out, ind) << "Then:", ind + 1);
      Else->dump(indent(out, ind) << "Else:", ind + 1);
      return out;
    }
  };

  /// ForExprAST - Expression class for for/in.
  class ForExprAST : public ExprAST {
    std::string VarName;
    std::unique_ptr<ExprAST> Start, End, Step, Body;

  public:
    ForExprAST(const std::string& VarName, std::unique_ptr<ExprAST> Start,
      std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
      std::unique_ptr<ExprAST> Body)
      : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
      Step(std::move(Step)), Body(std::move(Body)) {}
    Value* codegen() override;
    raw_ostream& dump(raw_ostream& out, int ind) override {
      ExprAST::dump(out << "for", ind);
      Start->dump(indent(out, ind) << "Cond:", ind + 1);
      End->dump(indent(out, ind) << "End:", ind + 1);
      Step->dump(indent(out, ind) << "Step:", ind + 1);
      Body->dump(indent(out, ind) << "Body:", ind + 1);
      return out;
    }
  };

  /// VarExprAST - Expression class for var/in
  class VarExprAST : public ExprAST {
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::unique_ptr<ExprAST> Body;

  public:
    VarExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::unique_ptr<ExprAST> Body)
      : VarNames(std::move(VarNames)), Body(std::move(Body)) {}
    Value* codegen() override;
    raw_ostream& dump(raw_ostream& out, int ind) override {
      ExprAST::dump(out << "var", ind);
      for (const auto& NamedVar : VarNames)
        NamedVar.second->dump(indent(out, ind) << NamedVar.first << ':', ind + 1);
      Body->dump(indent(out, ind) << "Body:", ind + 1);
      return out;
    }
  };

  /// PrototypeAST - This class represents the "prototype" for a function,
  /// which captures its name, and its argument names (thus implicitly the number
  /// of arguments the function takes), as well as if it is an operator.
  class PrototypeAST {
    std::string Name;
    std::vector<std::string> Args;
    bool IsOperator;
    unsigned Precedence; // Precedence if a binary op.
    int Line;

  public:
    PrototypeAST(SourceLocation Loc, const std::string& Name,
      std::vector<std::string> Args, bool IsOperator = false,
      unsigned Prec = 0)
      : Name(Name), Args(std::move(Args)), IsOperator(IsOperator),
      Precedence(Prec), Line(Loc.Line) {}
    Function* codegen();
    const std::string& getName() const { return Name; }

    bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
    bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

    char getOperatorName() const {
      assert(isUnaryOp() || isBinaryOp());
      return Name[Name.size() - 1];
    }

    unsigned getBinaryPrecedence() const { return Precedence; }
    int getLine() const { return Line; }
  };

  /// FunctionAST - This class represents a function definition itself.
  class FunctionAST {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;

  public:
    FunctionAST(std::unique_ptr<PrototypeAST> Proto,
      std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
    Function* codegen();
    raw_ostream& dump(raw_ostream& out, int ind) {
      indent(out, ind) << "FunctionAST\n";
      ++ind;
      indent(out, ind) << "Body:";
      return Body ? Body->dump(out, ind) : out << "null\n";
    }
    const PrototypeAST& getProto() const;
    const std::string& getName() const;
  };
} // end anonymous namespace


static llvm::orc::ThreadSafeModule irgenAndTakeOwnership(FunctionAST& FnAST, const std::string& Suffix);



namespace llvm {
  namespace orc {

    class KaleidoscopeASTLayer;
    class KaleidoscopeJIT;

    class KaleidoscopeASTMaterializationUnit : public MaterializationUnit {
    public:
      KaleidoscopeASTMaterializationUnit(KaleidoscopeASTLayer& L,
        std::unique_ptr<FunctionAST> F);

      StringRef getName() const override {
        return "KaleidoscopeASTMaterializationUnit";
      }

      void materialize(std::unique_ptr<MaterializationResponsibility> R) override;

    private:
      void discard(const JITDylib& JD, const SymbolStringPtr& Sym) override {
        llvm_unreachable("Kaleidoscope functions are not overridable");
      }

      KaleidoscopeASTLayer& L;
      std::unique_ptr<FunctionAST> F;
    };

    class KaleidoscopeASTLayer {
    public:
      KaleidoscopeASTLayer(IRLayer& BaseLayer, const DataLayout& DL)
        : BaseLayer(BaseLayer), DL(DL) {}

      Error add(ResourceTrackerSP RT, std::unique_ptr<FunctionAST> F) {
        return RT->getJITDylib().define(
          std::make_unique<KaleidoscopeASTMaterializationUnit>(*this,
            std::move(F)),
          RT);
      }

      void emit(std::unique_ptr<MaterializationResponsibility> MR,
        std::unique_ptr<FunctionAST> F) {
       // BaseLayer.emit(std::move(MR), irgenAndTakeOwnership(*F, ""));
      }

      MaterializationUnit::Interface getInterface(FunctionAST& F) {
        MangleAndInterner Mangle(BaseLayer.getExecutionSession(), DL);
        SymbolFlagsMap Symbols;
        Symbols[Mangle(F.getName())] =
          JITSymbolFlags(JITSymbolFlags::Exported | JITSymbolFlags::Callable);
        return MaterializationUnit::Interface(std::move(Symbols), nullptr);
      }

    private:
      IRLayer& BaseLayer;
      const DataLayout& DL;
    };

    KaleidoscopeASTMaterializationUnit::KaleidoscopeASTMaterializationUnit(
      KaleidoscopeASTLayer& L, std::unique_ptr<FunctionAST> F)
      : MaterializationUnit(L.getInterface(*F)), L(L), F(std::move(F)) {}

    void KaleidoscopeASTMaterializationUnit::materialize(
      std::unique_ptr<MaterializationResponsibility> R) {
      L.emit(std::move(R), std::move(F));
    }

    class KaleidoscopeJIT {
    private:
      std::unique_ptr<ExecutionSession> ES;
      std::unique_ptr<EPCIndirectionUtils> EPCIU;

      DataLayout DL;
      MangleAndInterner Mangle;

      RTDyldObjectLinkingLayer ObjectLayer;
      IRCompileLayer CompileLayer;
      IRTransformLayer OptimizeLayer;
      KaleidoscopeASTLayer ASTLayer;

      JITDylib& MainJD;

      static void handleLazyCallThroughError() {
        errs() << "LazyCallThrough error: Could not find function body";
        exit(1);
      }

    public:
      KaleidoscopeJIT(std::unique_ptr<ExecutionSession> ES,
        std::unique_ptr<EPCIndirectionUtils> EPCIU,
        JITTargetMachineBuilder JTMB, DataLayout DL)
        : ES(std::move(ES)), EPCIU(std::move(EPCIU)), DL(std::move(DL)),
        Mangle(*this->ES, this->DL),
        ObjectLayer(*this->ES,
          []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(*this->ES, ObjectLayer,
          std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
        OptimizeLayer(*this->ES, CompileLayer, optimizeModule),
        ASTLayer(OptimizeLayer, this->DL),
        MainJD(this->ES->createBareJITDylib("<main>")) {
        MainJD.addGenerator(
          cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL.getGlobalPrefix())));
      }

      ~KaleidoscopeJIT() {
        if (auto Err = ES->endSession())
          ES->reportError(std::move(Err));
        if (auto Err = EPCIU->cleanup())
          ES->reportError(std::move(Err));
      }

      static Expected<std::unique_ptr<KaleidoscopeJIT>> Create() {
        auto EPC = SelfExecutorProcessControl::Create();
        if (!EPC)
          return EPC.takeError();

        auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));

        auto EPCIU = EPCIndirectionUtils::Create(*ES);
        if (!EPCIU)
          return EPCIU.takeError();

        (*EPCIU)->createLazyCallThroughManager(
          *ES, ExecutorAddr::fromPtr(&handleLazyCallThroughError));

        if (auto Err = setUpInProcessLCTMReentryViaEPCIU(**EPCIU))
          return std::move(Err);

        JITTargetMachineBuilder JTMB(
          ES->getExecutorProcessControl().getTargetTriple());

        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL)
          return DL.takeError();

        return std::make_unique<KaleidoscopeJIT>(std::move(ES), std::move(*EPCIU),
          std::move(JTMB), std::move(*DL));
      }

      const DataLayout& getDataLayout() const { return DL; }

      JITDylib& getMainJITDylib() { return MainJD; }

      Error addModule(ThreadSafeModule TSM, ResourceTrackerSP RT = nullptr) {
        if (!RT)
          RT = MainJD.getDefaultResourceTracker();

        return OptimizeLayer.add(RT, std::move(TSM));
      }

      Error addAST(std::unique_ptr<FunctionAST> F, ResourceTrackerSP RT = nullptr) {
        if (!RT)
          RT = MainJD.getDefaultResourceTracker();
        return ASTLayer.add(RT, std::move(F));
      }

      Expected<ExecutorSymbolDef> lookup(StringRef Name) {
        return ES->lookup({ &MainJD }, Mangle(Name.str()));
      }

    private:
      static Expected<ThreadSafeModule>
        optimizeModule(ThreadSafeModule TSM, const MaterializationResponsibility& R) {
        TSM.withModuleDo([](Module& M) {
          // Create a function pass manager.
          auto FPM = std::make_unique<legacy::FunctionPassManager>(&M);

          // Add some optimizations.
          FPM->add(createInstructionCombiningPass());
          FPM->add(createReassociatePass());
          FPM->add(createGVNPass());
          FPM->add(createCFGSimplificationPass());
          FPM->doInitialization();

          // Run the optimizations over all functions in the module being added to
          // the JIT.
          for (auto& F : M)
            FPM->run(F);
          });

        return std::move(TSM);
      }
    };

  } // end namespace orc
} // end namespace llvm

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok = 0;
static int getNextToken() { return CurTok = gettok(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}

/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char* Str) {
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char* Str) {
  LogError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("expected ')'");
  getNextToken(); // eat ).
  return V;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  SourceLocation LitLoc = CurLoc;

  getNextToken(); // eat identifier.

  if (CurTok != '(') // Simple variable ref.
    return std::make_unique<VariableExprAST>(LitLoc, IdName);

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return std::make_unique<CallExprAST>(LitLoc, IdName, std::move(Args));
}

/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr() {
  SourceLocation IfLoc = CurLoc;

  getNextToken(); // eat the if.

  // condition.
  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != tok_then)
    return LogError("expected then");
  getNextToken(); // eat the then

  auto Then = ParseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != tok_else)
    return LogError("expected else");

  getNextToken();

  auto Else = ParseExpression();
  if (!Else)
    return nullptr;

  return std::make_unique<IfExprAST>(IfLoc, std::move(Cond), std::move(Then),
    std::move(Else));
}

/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {
  getNextToken(); // eat the for.

  if (CurTok != tok_identifier)
    return LogError("expected identifier after for");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  if (CurTok != '=')
    return LogError("expected '=' after for");
  getNextToken(); // eat '='.

  auto Start = ParseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("expected ',' after for start value");
  getNextToken();

  auto End = ParseExpression();
  if (!End)
    return nullptr;

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == ',') {
    getNextToken();
    Step = ParseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != tok_in)
    return LogError("expected 'in' after for");
  getNextToken(); // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
    std::move(Step), std::move(Body));
}

/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr() {
  getNextToken(); // eat the var.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("expected identifier after var");

  while (true) {
    std::string Name = IdentifierStr;
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseExpression();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("expected identifier list after var");
  }

  // At this point, we have to have 'in'.
  if (CurTok != tok_in)
    return LogError("expected 'in' keyword after 'var'");
  getNextToken(); // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<VarExprAST>(std::move(VarNames), std::move(Body));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch (CurTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
  case tok_var:
    return ParseVarExpr();
  case tok_eof:
    return nullptr;
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary() {
  // If the current token is not an operator, it must be a primary expr.
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
    return ParsePrimary();

  // If this is a unary operator, read it.
  int Opc = CurTok;
  getNextToken();
  if (auto Operand = ParseUnary())
    return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}

/// binoprhs
///   ::= ('+' unary)*

static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
  std::unique_ptr<ExprAST> LHS) {
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok;
    SourceLocation BinLoc = CurLoc;
    getNextToken(); // eat binop

    // Parse the unary expression after the binary operator.
    auto RHS = ParseUnary();
    if (!RHS)
      return nullptr;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = GetTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS.
    LHS = std::make_unique<BinaryExprAST>(BinLoc, BinOp, std::move(LHS),
      std::move(RHS));
  }
}

/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParseUnary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  std::string FnName;

  SourceLocation FnLoc = CurLoc;

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Expected function name in prototype");
  case tok_identifier:
    FnName = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected unary operator");
    FnName = "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected binary operator");
    FnName = "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return LogErrorP("Invalid precedence: must be 1..100");
      BinaryPrecedence = (unsigned)NumVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Invalid number of operands for operator");

  return std::make_unique<PrototypeAST>(FnLoc, FnName, ArgNames, Kind != 0,
    BinaryPrecedence);
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

/// toplevelexpr ::= expression
static std::vector<std::unique_ptr<ExprAST>> ParseTopLevelExpr() {
  SourceLocation FnLoc = CurLoc;
  std::vector<std::unique_ptr<ExprAST>> Expressions;

  while (auto E = ParseExpression()) {
    Expressions.push_back(std::move(E));

    if (CurTok == ';')
      getNextToken(); // eat the semicolon
    else
      break;
  }
  return Expressions;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}

//===----------------------------------------------------------------------===//
// Code Generation Globals
//===----------------------------------------------------------------------===//

static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<IRBuilder<>> Builder;
static ExitOnError ExitOnErr;

static std::map<std::string, AllocaInst*> NamedValues;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

//===----------------------------------------------------------------------===//
// Debug Info Support
//===----------------------------------------------------------------------===//

static std::unique_ptr<DIBuilder> DBuilder;

DIType* DebugInfo::getDoubleTy() {
  if (DblTy)
    return DblTy;

  DblTy = DBuilder->createBasicType("double", 64, dwarf::DW_ATE_float);
  return DblTy;
}

void DebugInfo::emitLocation(ExprAST* AST) {
  if (!AST)
    return Builder->SetCurrentDebugLocation(DebugLoc());
  DIScope* Scope;
  if (LexicalBlocks.empty())
    Scope = TheCU;
  else
    Scope = LexicalBlocks.back();
  Builder->SetCurrentDebugLocation(DILocation::get(
    Scope->getContext(), AST->getLine(), AST->getCol(), Scope));
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
  case '<':
    L = Builder->CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
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

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock* LoopBB = BasicBlock::Create(*TheContext, "loop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(LoopBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  AllocaInst* OldVal = NamedValues[VarName];
  NamedValues[VarName] = Alloca;

  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  if (!Body->codegen())
    return nullptr;

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

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value* CurVar = Builder->CreateLoad(Type::getDoubleTy(*TheContext), Alloca,
    VarName.c_str());
  Value* NextVar = Builder->CreateFAdd(CurVar, StepVal, "nextvar");
  Builder->CreateStore(NextVar, Alloca);

  // Convert condition to a bool by comparing non-equal to 0.0.
  EndCond = Builder->CreateFCmpONE(
    EndCond, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");

  // Create the "after loop" block and insert it.
  BasicBlock* AfterBB =
    BasicBlock::Create(*TheContext, "afterloop", TheFunction);

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
  // Make the function type:  double(double,double) etc.
  std::vector<Type*> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
  FunctionType* FT =
    FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);

  Function* F =
    Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto& Arg : F->args())
    Arg.setName(Args[Idx++]);

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


//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void InitializeModule() {
  // Open a new module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());

  Builder = std::make_unique<IRBuilder<>>(*TheContext);
}

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (!FnAST->codegen())
      fprintf(stderr, "Error reading function definition:");
  }
  else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (!ProtoAST->codegen())
      fprintf(stderr, "Error reading extern");
    else
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
  }
  else {
    // Skip token for error recovery.
    getNextToken();
  }
}

class CompoundExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Expressions;

public:
  CompoundExprAST(std::vector<std::unique_ptr<ExprAST>> Expressions)
    : Expressions(std::move(Expressions)) {}

  Value* codegen() override {
    for (auto& Expr : Expressions) {
      if (!Expr->codegen())
        return nullptr;
    }
    return Constant::getNullValue(Type::getDoubleTy(*TheContext));
  }
};

static void HandleTopLevelExpression() {
  // Evaluate all top-level expressions
  auto expressions = ParseTopLevelExpr();

  // Create a prototype for 'main'
  auto Proto = std::make_unique<PrototypeAST>(
    CurLoc, "main", std::vector<std::string>());

  // Combine expressions into a single body
  std::vector<std::unique_ptr<ExprAST>> BodyExpressions;
  for (auto& expr : expressions) {
    BodyExpressions.push_back(std::move(expr));
  }
  auto Body = std::make_unique<CompoundExprAST>(std::move(BodyExpressions));

  auto fn_ast = std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  if (fn_ast) {
    if (!fn_ast->codegen()) {
      fprintf(stderr, "Error generating code for top level expr\n");
    }
  }
  else {
    // Skip token for error recovery.
    getNextToken();
  }
}


/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    switch (CurTok) {
    case tok_eof:
      code_input.clear(); // clear buffer
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double x) {
  fputc((char)x, stderr);
  return 0;
}

std::vector<loco_t::shape_t> shapes;

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double x) {
  
  fan::printcl(x);
  //fprintf(stderr, "%f\n", X);
  return 0;
}

//extern "C" DLLEXPORT double rectangle(double px, double py) {
//  shapes.push_back(fan::graphics::rectangle_t{ {
//      .position = fan::vec2(px, py),
//      .size = fan::vec2(50, 50),
//      .color = fan::random::color()
//} });
//  return 0;
//}

extern "C" DLLEXPORT double rectangle(double px, double py, double sx, double sy) {
  shapes.push_back(fan::graphics::rectangle_t{ {
      .position = fan::vec2(px, py),
      .size = fan::vec2(sx, sx),
      .color = fan::random::color()
} });
  return 0;
}

extern "C" DLLEXPORT double sleep(double x) {
  std::this_thread::sleep_for(std::chrono::duration<double>(x));
  return 0;
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
#include "llvm/MC/TargetRegistry.h"




struct pile_t {
  
  loco_t loco;
  event_loop_t event_loop;
}pile;

void init() {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  KSDbgInfo.TheCU = nullptr;
  KSDbgInfo.DblTy = nullptr;
  KSDbgInfo.LexicalBlocks = {};

  TheContext = std::unique_ptr<LLVMContext>{};
  TheModule = std::unique_ptr<Module>{};
  Builder = std::unique_ptr<IRBuilder<>>{};
  ExitOnErr = ExitOnError{};
  NamedValues = std::map<std::string, AllocaInst*>{};
  TheJIT = std::unique_ptr<KaleidoscopeJIT>{};
  FunctionProtos = std::map<std::string, std::unique_ptr<PrototypeAST>>{};
  DBuilder = std::unique_ptr<DIBuilder>{};

  BinopPrecedence = std::map<char, int>{};
  IdentifierStr = std::string(); // Filled in if tok_identifier
  NumVal = double();             // Filled in if tok_number

  CurTok = 0;
  gLastChar = ' ';

  CurLoc = SourceLocation();
  LexLoc = SourceLocation{ 1, 0 };


  // Install standard binary operators. 1 is lowest precedence.
  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  getNextToken();
  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();

  // Add the current debug info version into the module.
  TheModule->addModuleFlag(Module::Warning, "Debug Info Version", DEBUG_METADATA_VERSION);

  // Darwin only supports dwarf2.
  if (Triple(sys::getProcessTriple()).isOSDarwin())
    TheModule->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);

  // Construct the DIBuilder, we do this here because we need the module.
  DBuilder = std::make_unique<DIBuilder>(*TheModule);

  // Create the compile unit for the module.
  // Currently down as "fib.ks" as a filename since we're redirecting stdin
  // but we'd like actual source locations.
  KSDbgInfo.TheCU = DBuilder->createCompileUnit(
    dwarf::DW_LANG_C, DBuilder->createFile("fib.ks", "."), "Kaleidoscope Compiler", false, "", 0);

  {
    std::vector<std::string> Args{ "x" }; // parameter names
    std::vector<Type*> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
    FunctionType* FT =
      FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);

    Function* F =
      Function::Create(FT, Function::ExternalLinkage, "printd", TheModule.get());

    // Set names for all arguments.
    unsigned Idx = 0;
    for (auto& Arg : F->args())
      Arg.setName(Args[Idx++]);

    //FunctionProtos[F->getName().str()] = std::move(F);
  }
}

void printDebugInfo(llvm::Module& M) {

  
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  M.print(OS, nullptr);  // Print module to string

  fan::printcl(OS.str());
}

void recompile() {
  
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  //// Run the main "interpreter loop" now.
  MainLoop();

  //parse_input();

  // Finalize the debug info.
  DBuilder->finalize();

  printDebugInfo(*TheModule);

  
  // Print out all of the generated code.
  //TheModule->print(errs(), nullptr);

  auto TargetTriple = sys::getDefaultTargetTriple();
  TheModule->setTargetTriple(TargetTriple);
  std::string Error;
  auto Target = TargetRegistry::lookupTarget(TargetTriple, Error);

  // Print an error and exit if we couldn't find the requested target.
  // This generally occurs if we've forgotten to initialize the
  // TargetRegistry or we have a bogus target triple.
  if (!Target) {
    errs() << Error;
    return;
  }

  auto CPU = "generic";
  auto Features = "";
  TargetOptions opt;
  auto TheTargetMachine = Target->createTargetMachine(TargetTriple, CPU, Features, opt, Reloc::PIC_);
  TheModule->setDataLayout(TheTargetMachine->createDataLayout());

  auto Filename = "output.o";
  std::error_code EC;
  raw_fd_ostream dest(Filename, EC, sys::fs::OF_None);
  if (EC) {
    errs() << "Could not open file: " << EC.message();
    return;
  }

  legacy::PassManager pass;
  auto FileType = CodeGenFileType::ObjectFile;
  if (TheTargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
    errs() << "TheTargetMachine can't emit a file of this type";
    return;
  }

  pass.run(*TheModule);
  dest.flush();
  outs() << "Wrote " << Filename << "\n";
}

double test(double x) {
  fan::print(x);
  return 0;
}

int run() {

  auto start = std::chrono::steady_clock::now();

  // Create the JIT engine and move the module into it
  std::string ErrorStr;
  std::unique_ptr<ExecutionEngine> EE(
    EngineBuilder(std::move(TheModule))
    .setErrorStr(&ErrorStr)
    .setOptLevel(CodeGenOptLevel::None)
    .create()
  );

  if (!EE) {
    errs() << "Failed to create ExecutionEngine: " << ErrorStr << "\n";
    return 1;
  }

  EE->finalizeObject();
  std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - start).count() << "ms\n";

  // Assuming you have a function named 'main' to execute
  Function* MainFn = EE->FindFunctionNamed("main");
  if (!MainFn) {
    errs() << "'main' function not found in module.\n";
    return 1;
  }

  std::vector<GenericValue> NoArgs;
  GenericValue GV = EE->runFunction(MainFn, NoArgs);

//  std::stringstream oss;
 // oss << "Result: " << std::fixed << std::setprecision(0) << GV.DoubleVal << std::endl;
  // Print the result
  fan::printcl("result: ", GV.DoubleVal);
  //std::cout << "Result: " << std::fixed << std::setprecision(0) << GV.DoubleVal << std::endl;
  return 0;
}
int main() {

  pile.loco.console.commands.add("clear_shapes", [](const fan::commands_t::arg_t& args) {
    shapes.clear();
  }).description = "";


  //task_t main_f = [&]() -> task_t {
  //  pile.loco.process_loop([] {
  //    ImGui::Begin("window");
  //    ImGui::Text("Hello, world!");
  //    if (ImGui::Button("Click me")) {
  //      // Do something
  //    }
  //    ImGui::End();
  //    
  //  });
  //  co_return;
  //}();
  //main_f.start();
  //while (1) {
  //  pile.event_loop.run_one();
  //}

  /*-------------------------------------------------*/

  TextEditor editor, input;

  TextEditor::LanguageDefinition lang = TextEditor::LanguageDefinition::CPlusPlus();
  // set your own known preprocessor symbols...
  static const char* ppnames[] = { "NULL" };
  // ... and their corresponding values
  static const char* ppvalues[] = {
    "#define NULL ((void*)0)",
  };

  for (int i = 0; i < sizeof(ppnames) / sizeof(ppnames[0]); ++i)
  {
    TextEditor::Identifier id;
    id.mDeclaration = ppvalues[i];
    lang.mPreprocIdentifiers.insert(std::make_pair(std::string(ppnames[i]), id));
  }

  //for (auto& i : commands.func_table) {
  //  TextEditor::Identifier id;
  //  id.mDeclaration = i.second.description;
  //  lang.mIdentifiers.insert(std::make_pair(i.first, id));
  //}

  editor.SetLanguageDefinition(lang);
  //

  auto palette = editor.GetPalette();

  palette[(int)TextEditor::PaletteIndex::Background] = 0xff202020;
  editor.SetPalette(palette);
  editor.SetPalette(editor.GetRetroBluePalette());
  editor.SetTabSize(2);
  editor.SetShowWhitespaces(false);

  auto file_name = "test.fpp";

  fan::string str;
  fan::io::file::read(
    file_name,
    &str
  );

  editor.SetText(str);

  int current_font = 2;

  bool block_zoom[2]{};

  float font_scale_factor = 1.0f;
  pile.loco.window.add_buttons_callback([&](const auto& d) {
    if (d.state != fan::mouse_state::press) {
      return;
    }
    if (pile.loco.window.key_pressed(fan::key_left_control) == false) {
      return;
    }

    auto& io = ImGui::GetIO();
    switch (d.button) {
    case fan::mouse_scroll_up: {
      if (block_zoom[0] == true) {
        break;
      }
      font_scale_factor *= 1.1;
      block_zoom[1] = false;
      break;
    }
    case fan::mouse_scroll_down: {
      if (block_zoom[1] == true) {
        break;
      }
      font_scale_factor *= 0.9;
      block_zoom[0] = false;
      break;
    }
    }

    //ImFont* selected_font = nullptr;
    //for (int i = 0; i < std::size(loco.fonts); ++i) {
    //  if (new_font_size <= font_size * (1 << i) / 2) {
    //    selected_font = loco.fonts[i];
    //    break;
    //  }
    //}

    if (font_scale_factor > 1.5) {
      current_font++;
      if (current_font > std::size(pile.loco.fonts) - 1) {
        current_font = std::size(pile.loco.fonts) - 1;
        block_zoom[0] = true;
      }
      else {
        io.FontDefault = pile.loco.fonts[current_font];
        font_scale_factor = 1;
      }
    }

    if (font_scale_factor < 0.5) {
      current_font--;
      if (current_font < 0) {
        current_font = 0;
        block_zoom[1] = true;
      }
      else {
        io.FontDefault = pile.loco.fonts[current_font];
        font_scale_factor = 1;
      }
    }


    // Set the window font scale
    io.FontGlobalScale = font_scale_factor;
    // Set the selected font for ImGui
    //io.FontDefault = selected_font;
    return;
  });

  /*-------------------------------------------------*/

  pile.loco.input_action.add_keycombo({ fan::key_left_control, fan::key_s }, "save_file");
  pile.loco.input_action.add_keycombo({ fan::key_f5 }, "compile_and_run");


  pile.loco.loop([&] {
    ImGui::Begin("window");
    if (ImGui::Button("compile")) {
      init();
      recompile();
      code_input = editor.GetText();
    }
    ImGui::SameLine();
    if (ImGui::Button("run")) {
      run();
      code_input.push_back(EOF);
    }
    ImGui::SameLine();
    if (ImGui::Button("compile & run")) {
      code_input = editor.GetText();
      code_input.push_back(EOF);

      init();
      recompile();
      run();
    }
    if (pile.loco.input_action.is_active("compile_and_run")) {
      code_input = editor.GetText();
      code_input.push_back(EOF);

      init();
      recompile();
      run();
    }
    editor.Render("editor");
    ImGui::End();
    if (pile.loco.input_action.is_active("save_file")) {
      std::string str = editor.GetText();
      fan::io::file::write(file_name, str.substr(0, std::max(0ull, str.size() - 1)), std::ios_base::binary);
    }

  });
}