#pragma once

namespace llvm {
  struct Module;
}

void init_code();
void recompile_code();
int run_code();

void printDebugInfo(llvm::Module& M);