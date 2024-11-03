#pragma once

#include <map>

#include "ast.h"

using namespace ast;

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
static std::map<char, int> BinopPrecedence{
  {'=', 2},
  {'<', 10},
  {'>', 10},
  {'+', 20},
  {'-', 20},
  {'*', 40},
  {'/', 40},
  { '%', 40 },
};

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
static std::unique_ptr<ExprAST> LogError(const char* Str) {
  fprintf(stderr, "Error: %s %d %d\n", Str, CurLoc.Line, CurLoc.Col);
  return nullptr;
}

static std::unique_ptr<PrototypeAST> LogErrorP(const char* Str) {
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

  // Parse compound body
  std::unique_ptr<ExprAST> Body;
  if (CurTok == '{') {
    getNextToken(); // eat '{'

    std::vector<std::unique_ptr<ExprAST>> Statements;
    // Parse expressions until we hit closing brace
    while (CurTok != '}' && CurTok != tok_eof) {
      auto E = ParseExpression();
      if (!E)
        return nullptr;
      Statements.push_back(std::move(E));

      // Allow optional semicolon
      if (CurTok == ';')
        getNextToken(); // eat ';'
    }

    if (CurTok != '}')
      return LogError("expected '}' after compound expression");
    getNextToken(); // eat '}'

    // Create compound expression containing all statements
    Body = std::make_unique<CompoundExprAST>(std::move(Statements));
  }
  else {
    // Single expression body
    Body = ParseExpression();
  }

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
  case tok_string: {
    auto Result = std::make_unique<StringExprAST>(StringVal);
    getNextToken();
    return Result;
  }
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
  unsigned Kind = 0;
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
  std::vector<std::string> ArgTypes;
  getNextToken(); // eat '('

  while (CurTok != ')') {
    std::string ArgType;

    if (CurTok == tok_type_string) {
      ArgType = "string";  // Recognize string type
    }
    else if (CurTok == tok_type_double) {
      ArgType = "double";  // Recognize double type
    }
    else if (CurTok == tok_identifier) {
      ArgNames.push_back(IdentifierStr);  // Store the argument name
      ArgTypes.push_back("double");  // Store the argument type
      getNextToken();
      if (CurTok == ',') getNextToken(); // Eat the comma and continue
      else if (CurTok == ')') break;
      else if (CurTok == tok_identifier) continue;
    }
    else {
      return LogErrorP("Expected type specifier before argument name");
    }

    getNextToken();  // Move to argument name

    if (CurTok != tok_identifier)
      return LogErrorP("Expected argument name");

    ArgNames.push_back(IdentifierStr);  // Store the argument name
    ArgTypes.push_back(ArgType);  // Store the argument type
    getNextToken();  // Move to the next token

    if (CurTok == ',') {
      getNextToken();  // Eat the comma
    }
    else if (CurTok == tok_identifier) continue;
  }

  if (CurTok != ')') {
    return LogErrorP("Expected ',' or ')' in argument list");
  }
  getNextToken();  // eat ')'

  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Invalid number of operands for operator");

  return std::make_unique<PrototypeAST>(FnLoc, FnName, ArgNames, ArgTypes, Kind != 0, BinaryPrecedence);
}






/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken();  // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  // Parse function body
  std::unique_ptr<ExprAST> Body;
  if (CurTok == '{') {
    getNextToken();  // eat '{'

    std::vector<std::unique_ptr<ExprAST>> Statements;
    while (CurTok != '}' && CurTok != tok_eof) {
      if (auto E = ParseExpression()) {
        Statements.push_back(std::move(E));
        // Allow optional semicolons between statements
        if (CurTok == ';')
          getNextToken();
      }
      else {
        return nullptr;
      }
    }

    if (CurTok != '}') {
      LogError("Expected '}' in function body");
      return nullptr;
    }
    getNextToken();  // eat '}'

    Body = std::make_unique<CompoundExprAST>(std::move(Statements));
  }
  else {
    Body = ParseExpression();
  }

  if (!Body)
    return nullptr;

  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
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