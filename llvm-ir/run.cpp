#include <pch.h>
#include "run.h"

#include <iostream>
#include <memory>
#include <vector>
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/LegacyPassManager.h"

#include "parser.h"
#include "codegen.h"

using namespace llvm;

extern DebugInfo KSDbgInfo;
extern std::unique_ptr<llvm::Module> TheModule;

void init_code() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  KSDbgInfo.TheCU = nullptr;
  KSDbgInfo.DblTy = nullptr;
  KSDbgInfo.LexicalBlocks = {};

  TheContext = std::unique_ptr<LLVMContext>{};
  TheModule = std::unique_ptr<Module>{};

  codegen_init();

  IdentifierStr = std::string(); // Filled in if tok_identifier
  NumVal = double();             // Filled in if tok_number

  CurTok = 0;
  gLastChar = ' ';

  CurLoc = SourceLocation();
  LexLoc = SourceLocation{ 1, 0 };

  // Prime the first token.
  getNextToken();

  create_the_JIT();

  init_module();

  // Add the current debug info version into the module.
  TheModule->addModuleFlag(Module::Warning, "Debug Info Version", DEBUG_METADATA_VERSION);

  // Darwin only supports dwarf2.
  if (Triple(llvm::sys::getProcessTriple()).isOSDarwin())
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
  }
}

int run_code() {
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

  //{
  //  using namespace object;
  //  std::string objectFileName("a.o");

  //  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer =
  //    MemoryBuffer::getFile(objectFileName.c_str());

  //  if (!buffer) {
  //    fan::throw_error("failed to open file");
  //  }

  //  Expected<std::unique_ptr<ObjectFile>> objectOrError =
  //    ObjectFile::createObjectFile(buffer.get()->getMemBufferRef());

  //  if (!objectOrError) {
  //    fan::throw_error("failed to open file");
  //  }

  //  std::unique_ptr<ObjectFile> objectFile(std::move(objectOrError.get()));

  //  auto owningObject = OwningBinary<ObjectFile>(std::move(objectFile),
  //    std::move(buffer.get()));

  //  EE->addObjectFile(std::move(owningObject));
  //}

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
    //fan::printcl("result: ", GV.DoubleVal);
    //std::cout << "Result: " << std::fixed << std::setprecision(0) << GV.DoubleVal << std::endl;
}


//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (!FnAST->codegen())
      fprintf(stderr, "Error reading function definition:");
  }
  else {//
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (!ProtoAST->codegen())
      fprintf(stderr, "Error reading extern");
    else
      get_function_protos()[ProtoAST->getName()] = std::move(ProtoAST);
  }
  else {
    // Skip token for error recovery.
    getNextToken();
  }
}


static void HandleTopLevelExpression() {
  // Evaluate all top-level expressions
  auto expressions = ParseTopLevelExpr();

  // Create a prototype for 'main'
  auto Proto = std::make_unique<PrototypeAST>(
    CurLoc, "main", std::vector<std::string>(), std::vector<std::string>());

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
void MainLoop() {
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


void recompile_code() {
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