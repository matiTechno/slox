#include "Array.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h> // check if stdin is terminal or pipe
#include <signal.h> // keyboard interrupt
#include <time.h>
#include <math.h>

// @TODO:
// closures, classes, inheritance
// api to call interpreter from code
// execute() and executeFile() functions to execute lox from lox
// there is no need to store variable names in byteArray, use pointers to source

bool isDigit(const char c) {return c >= '0' && c <= '9';}

bool isAlphanumeric(const char c)
{
    return isDigit(c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

template<typename T>
T min(T l, T r) {return l > r ? r : l;}

template<typename T, int N>
int getSize(T(&)[N]) {return N;}

enum class TokenType
{
    LOX_EOF = 0,

    // single-character tokens
    LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE,
    COMMA, DOT, MINUS, PLUS, SEMICOLON, SLASH, STAR,

    // one or two character tokens
    BANG, BANG_EQUAL,
    EQUAL, EQUAL_EQUAL,
    GREATER, GREATER_EQUAL,
    LESS, LESS_EQUAL,

    // literls
    IDENTIFIER, STRING, NUMBER,

    // keywords
    AND, CLASS, ELSE, FALSE, FUN, FOR, IF, NIL, OR,
    PRINT, RETURN, SUPER, THIS, TRUE, VAR, WHILE
};

struct Keyword
{
    const char* str;
    TokenType tokenType;
};

static Keyword _keywords[] = {
    {"and", TokenType::AND}, {"class", TokenType::CLASS}, {"else", TokenType::ELSE},
    {"false", TokenType::FALSE}, {"fun", TokenType::FUN}, {"for", TokenType::FOR},
    {"if", TokenType::IF}, {"nil", TokenType::NIL}, {"or", TokenType::OR},
    {"print", TokenType::PRINT}, {"return", TokenType::RETURN}, {"super", TokenType::SUPER},
    {"this", TokenType::THIS}, {"var", TokenType::VAR}, {"while", TokenType::WHILE},
    {"true", TokenType::TRUE}
};

struct Token
{
    TokenType type;
    int col;
    int line;

    union
    {
        double number;

        struct
        {
            const char* begin;
            int len;
        } string; // string literal or identifier
    };
};

enum class ValueType
{
    BOOLEAN,
    NUMBER,
    STRING,
    NIL,
    CALLABLE,
};

struct Stmt;

struct Value
{
    ValueType type;

    union
    {
        bool boolean;
        double number;
        const char* string;

        struct
        {
            int numParams;
            // token is for error reporting
            bool (*native)(Value& value, Token token, const Value* args);

            // use these if native is nullptr
            // @TODO:
            // we can use pointers because statements will not grow during value lifetime
            // oh, that's not true, what about repl sessions or executing new script from 
            // current script
            const Stmt* stmtBegin;
            const Stmt* stmtEnd;
            const char* paramStr; // these are parameters names separated by '\0'
        } callable;
    };
};

enum class ExprType
{
    BINARY,
    GROUPING,
    PRIMARY,
    UNARY,
    CALL
};

enum
{
    // yes, LOX supports only 8 function arguments
    MAXARGS = 8
};

struct Expr
{
    ExprType type;

    union
    {
        struct
        {
            int idxExprLeft;
            int idxExprRight;
            Token operatorToken;
        } binary;

        struct
        {
            int idxExpr;
        } grouping; // we need this to prevent statements like '(myvar) = 5;' from passing

        struct
        {
            int idxExpr;
            Token operatorToken;
        } unary;

        struct
        {
            Token token;
        } primary;

        struct
        {
            int idxExprCallee;
            int idxExprArg;
            int numArgs;
            Token token; // used for error reporting
        } call;
    };
};

struct Line
{
    const char* begin;
    int len;
};

struct Var
{
    const char* name;
    Value value;
};

struct Environment
{
    int idxVar;
    int count;
    bool canBeShadowed;
    int idxParentEnv;
};

enum class ErrorType
{
    LEXER,
    PARSER,
    RUNTIME
};

enum class StmtType
{
    EXPRESSION,
    PRINT,
    VAR_DECL,
    BLOCK,
    IF_STMT,
    WHILE_STMT,
    FUN_DECL,
    RETURN_STMT
};

struct Stmt
{
    StmtType type;

    union
    {
        struct
        {
            int idxExpr;
        } expression;

        struct
        {
            int idxExpr;
        } print;

        struct
        {
            Token identifierToken;
            int idxExpr;
        } varDecl;

        struct
        {
            int idxExpr;
            int count;
            int elseCount;
        } ifStmt;

        struct
        {
            int idxExpr;
            int count;
        } whileStmt;

        struct
        {
            Token identifierToken;
            const char* paramStr;
            int numParams;
            int count; // number of statements in a function body
        } funDecl;

        struct
        {
            int idxExpr;
        } returnStmt;

        struct
        {
            bool canBeShadowed;
            int count;
        } block;
    };
};

void printError(ErrorType errorType, int col, int line, const char* str);

struct Context
{
    Array<Line> lines;
    Array<Expr> expressions;
    char scratchBuf[10000];
    Array<Var> variables;
    Array<Environment> environments;

    // @TODO:
    // temporary, storage for string values
    char byteArray[10000];
    int end = 0;

    struct
    {
        // set only during VAR_DECL execution
        const char* begin = nullptr;
        int len;
    } newVarName;

    struct
    {
        bool success;
        Value value;
    } funReturn; // eh.

    Context()
    {
        // push global env; we can't use pushEnv() here
        environments.pushBack({0, 0, true});

        // add some native functions
        const char* name;
        Value value;
        value.type = ValueType::CALLABLE;

        {
            value.callable.numParams = 0;
            value.callable.native = [](Value& value, const Token token, const Value* args)
            {
                (void)token;
                (void)args;

                value.type = ValueType::NUMBER;

                timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                value.number = ts.tv_sec + ts.tv_nsec / 1000000000.0;
                return true;
            };

            name = "clock";
            addVar(name, strlen(name), value);
        }
        {
            value.callable.numParams = 1;
            value.callable.native = [](Value& value, const Token token, const Value* args)
            {
                if(args[0].type != ValueType::NUMBER)
                {
                    printError(ErrorType::RUNTIME, token.col, token.line,
                            "expected number as the argument");
                    return false;
                }

                value.type = ValueType::NUMBER;
                value.number = sin(args[0].number);
                return true;
            };

            name = "sin";
            addVar(name, strlen(name), value);
        }
    }

    void pushEnv(bool canBeShadowed) {pushEnv(canBeShadowed, environments.size() - 1);}

    void pushEnv(bool canBeShadowed, const int idxParentEnv)
    {
        environments.pushBack({environments.back().idxVar + environments.back().count, 0,
                canBeShadowed, idxParentEnv});
    }

    void  popEnv()
    {
        assert(environments.size() > 1);
        variables.erase(environments.back().idxVar, environments.back().count);
        environments.popBack();
    }

    bool addVar(const char* const name, const int len, Value value)
    {
        int idx = -1;
        for(int i = variables.size() - 1; i >= 0; --i)
        {
            if(int(strlen(variables[i].name)) == len &&
               strncmp(variables[i].name, name, len) == 0)
            {
                idx = i;
                break;
            }
        }

        // order is important

        if(idx == -1)
        {
            variables.pushBack( {addString(name, len), value} );
            environments.back().count += 1;
        }
        // we allow variable redeclaration for globals
        else if(environments.back().idxVar == 0)
        {
            variables[idx].value = value;
        }
        // variable is already defined but in upper scope
        else if(idx < environments.back().idxVar)
        {
            const Environment& parentEnv = environments[environments.back().idxParentEnv];

            if( (idx < parentEnv.idxVar) || parentEnv.canBeShadowed)
            {
                variables.pushBack( {addString(name, len), value} );
                environments.back().count += 1;
            }
            else
                return false;
        }
        else // variable is already declared in the current scope, error
            return false;

        return true;
    }

    Var* getVar(const char* const name, int len)
    {
        const Environment* env = environments.end() - 1;
        for(;;)
        {
            for(int i = env->idxVar; i < env->idxVar + env->count; ++i)
            {
                if(int(strlen(variables[i].name)) == len &&
                   strncmp(variables[i].name, name, len) == 0)
                {
                    return &variables[i];
                }
            }

            if(env == environments.begin())
                break;

            env = &environments[env->idxParentEnv];
        }
        return nullptr;
    }

    int addExpr(const Expr expr)
    {
        expressions.pushBack(expr);
        return expressions.size() - 1;
    }

    const char* addString(const char* const string, int len)
    {
        const int start = end;
        end += len + 1;

        assert(end <= int(sizeof(byteArray))); // @ cast

        memcpy(&byteArray[start], string, len);

        byteArray[end - 1] = '\0';

        return &byteArray[start];
    }

    const char* addString(const char* const lstr, const char* const rstr)
    {
        const int start = end;
        const int llen = strlen(lstr);
        const int rlen = strlen(rstr);

        end += llen + rlen + 1;

        assert(end <= int(sizeof(byteArray))); // @ cast

        memcpy(&byteArray[start], lstr, llen);
        memcpy(&byteArray[start + llen], rstr, rlen + 1);

        return &byteArray[start];
    }

} static _context;

void printError(ErrorType errorType, int col, int line, const char* str)
{
    printf("\n%.*s\n", _context.lines[line - 1].len, _context.lines[line - 1].begin);

    for(int i = 0; i < col - 1; ++i)
        putchar('-');

    const char* typeStr;

    switch(errorType)
    {
        case ErrorType::LEXER: typeStr = "lexer"; break;
        case ErrorType::PARSER: typeStr = "parser"; break;
        case ErrorType::RUNTIME: typeStr = "runtime"; break;
        default: assert(false);
    }

    printf("^\n%d:%d: {%s} %s\n\n", line, col, typeStr, str);
}

bool expression(Expr& expr, const Token** const token);

bool primary(Expr& outputExpr, const Token** const token)
{
    const TokenType type = (**token).type;

    if(type == TokenType::FALSE || type == TokenType::TRUE || type == TokenType::NIL ||
       type == TokenType::NUMBER || type == TokenType::STRING || type == TokenType::IDENTIFIER)
    {
        outputExpr.type = ExprType::PRIMARY;
        outputExpr.primary.token = **token;
        ++(*token);
        return true;
    }
    else if(type == TokenType::LEFT_PAREN)
    {
        ++(*token);
        Expr expr;
        if(!expression(expr, token)) return false;

        if((**token).type == TokenType::RIGHT_PAREN)
        {
            ++(*token);
            outputExpr.type = ExprType::GROUPING;
            outputExpr.grouping.idxExpr = _context.addExpr(expr);
            return true;
        }
        else
        {
            printError(ErrorType::PARSER, (**token).col, (**token).line,
                    "expected ')' after expression");
            return false;
        }
    }

    printError(ErrorType::PARSER, (**token).col, (**token).line, "expected expression");
    return false;
}

bool call(Expr& expr, const Token** const token)
{
    if(!primary(expr, token)) return false;

    while(true)
    {
        if((**token).type == TokenType::LEFT_PAREN)
        {
            expr.call.idxExprCallee = _context.addExpr(expr);
            expr.type = ExprType::CALL;
            expr.call.token = **token;
            expr.call.numArgs = 0;

            Expr args[MAXARGS];

            ++(*token);

            if((**token).type != TokenType::RIGHT_PAREN)
            {
                while(true)
                {
                    if(expr.call.numArgs == MAXARGS)
                    {
                        snprintf(_context.scratchBuf, getSize(_context.scratchBuf),
                                "LOX supports only %d function arguments", MAXARGS);

                        printError(ErrorType::PARSER, (**token).col, (**token).line,
                                _context.scratchBuf);
                        return false;
                    }

                    if(!expression(args[expr.call.numArgs], token)) return false;
                    ++expr.call.numArgs;

                    if((**token).type != TokenType::COMMA)
                        break;

                    ++(*token);
                }
            }

            // we can't add the arguments to _context.expressions in while(true)
            // because expression() can call _context.addExpr() itself and we want
            // arguments to be continuous in a vector

            for(int i = 0; i < expr.call.numArgs; ++i)
            {
                const int idx = _context.addExpr(args[i]);

                if(i == 0)
                    expr.call.idxExprArg = idx;
            }

            if((**token).type != TokenType::RIGHT_PAREN)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "exprected ')' after arguments");
                return false;
            }

            ++(*token);
        }
        else
            break;
    }

    return true;
}

bool unary(Expr& outputExpr, const Token** const token)
{
    if((**token).type == TokenType::BANG || (**token).type == TokenType::MINUS)
    {
        Token operatorToken = **token;
        ++(*token);

        Expr expr;
        if(!unary(expr, token)) return false;

        outputExpr.type = ExprType::UNARY;
        outputExpr.unary.idxExpr = _context.addExpr(expr);
        outputExpr.unary.operatorToken = operatorToken;
        return true;
    }

    return call(outputExpr, token);
}

enum class BinaryType
{
    LOGIC_OR,
    LOGIC_AND,
    EQUALITY,
    COMPARISON,
    ADDITION,
    MULTIPLICATION,
    GOTO_UNARY
};

// all these binary operators are left-associative
bool binary(Expr& expr, const Token** const token, BinaryType binaryType)
{
    TokenType tokenTypes[5] = {};

    switch(binaryType)
    {
        case BinaryType::GOTO_UNARY: return unary(expr, token);

        case BinaryType::LOGIC_OR:
        {
            tokenTypes[0] = TokenType::OR;
            break;
        }
        case BinaryType::LOGIC_AND:
        {
            tokenTypes[0] = TokenType::AND;
            break;
        }
        case BinaryType::EQUALITY:
        {
            tokenTypes[0] = TokenType::EQUAL_EQUAL;
            tokenTypes[1] = TokenType::BANG_EQUAL;
            break;
        }
        case BinaryType::COMPARISON:
        {
            tokenTypes[0] = TokenType::GREATER;
            tokenTypes[1] = TokenType::GREATER_EQUAL;
            tokenTypes[2] = TokenType::LESS;
            tokenTypes[3] = TokenType::LESS_EQUAL;
            break;
        }
        case BinaryType::ADDITION:
        {
            tokenTypes[0] = TokenType::PLUS;
            tokenTypes[1] = TokenType::MINUS;
            break;
        }
        case BinaryType::MULTIPLICATION:
        {
            tokenTypes[0] = TokenType::STAR;
            tokenTypes[1] = TokenType::SLASH;
            break;
        }
    }

    binaryType = BinaryType(int(binaryType) + 1);

    if(!binary(expr, token, binaryType)) return false;

    for(;;)
    {
        const TokenType* tokenType = tokenTypes;
        bool match = false;

        while(*tokenType != TokenType::LOX_EOF)
        {
            if(*tokenType == (**token).type)
            {
                match = true;
                break;
            }
            ++tokenType;
        }

        if(!match) break;

        const Token operatorToken = **token;
        ++(*token);

        Expr exprRight;

        if(!binary(exprRight, token, binaryType)) return false;

        // add expr to vector before modifying it
        expr.binary.idxExprLeft = _context.addExpr(expr);

        expr.type = ExprType::BINARY;
        expr.binary.idxExprRight = _context.addExpr(exprRight);
        expr.binary.operatorToken = operatorToken;
    }

    return true;
}

// assignment is right-associative (that's why we are using recursion instead of a loop)
bool assignment(Expr& expr, const Token** const token)
{
    if(!binary(expr, token, BinaryType::LOGIC_OR)) return false;

    if((**token).type == TokenType::EQUAL)
    {
        const Token operatorToken = **token;
        ++(*token);

        Expr exprRight;
        if(!assignment(exprRight, token)) return false;

        if(expr.type != ExprType::PRIMARY || expr.primary.token.type != TokenType::IDENTIFIER)
        {
            printError(ErrorType::PARSER, operatorToken.col, operatorToken.line,
                    "left side of the assignment must be lvalue");
            return false;
        }

        // add expr to vector before modifying it
        expr.binary.idxExprLeft = _context.addExpr(expr);

        expr.type = ExprType::BINARY;
        expr.binary.idxExprRight = _context.addExpr(exprRight);
        expr.binary.operatorToken = operatorToken;
    }

    return true;
}

bool expression(Expr& expr, const Token** const token)
{
    return assignment(expr, token);
}

struct Lexer
{
    Lexer(Array<Token>& tokens, const char* source): it(source), tokens(tokens) {}

    int line = 1;
    int col = 0;
    int tokenCol;
    int tokenLine;

    void addToken(TokenType type) {addTokenImpl(type);}

    void addToken(TokenType type, const char* string, int stringLen)
    {
        addTokenImpl(type, string, stringLen);
    }
    void addToken(TokenType type, double number)
    {
        addTokenImpl(type, nullptr, 0, number);
    }

    bool end() const {return *it == '\0';}

    char advance()
    {
        ++it;
        const char c = *(it - 1);

        if(c == '\n')
        {
            ++line;
            col = 0;
        }
        else
            ++col;

        return c;
    }

    bool match(char c)
    {
        if(end())
            return false;

        if(*it != c)
            return false;

        advance();
        return true;
    }

    const char* pos() const {return it - 1;} // in fact it is a previous position
    char peek() const {return *it;}
    char peekNext() const {return end() ? *it : *(it + 1);}

private:
    const char* it;
    Array<Token>& tokens;

    void addTokenImpl(TokenType type, const char* string = nullptr, int stringLen = 0,
            double number = 0.0)
    {
        Token token;
        token.type = type;
        token.col = tokenCol;
        token.line = tokenLine;

        if(string)
        {
            token.string.begin = string;
            token.string.len = stringLen;
        }
        else
            token.number = number;

        tokens.pushBack(token);
    }
};

// scanning and lexing mean the same thing
bool scan(Array<Token>& tokens, const char* const source)
{
    Lexer lexer(tokens, source);

    // fill _context.lines
    {
        const char* it = source;
        int lineLen = 0;
        const char* begin = it;
        while(*it != '\0')
        {
            if(*it == '\n')
            {
                _context.lines.pushBack({begin, lineLen});
                lineLen = 0;
                begin = it + 1;
            }
            else
                ++lineLen;

            ++it;
        }

        _context.lines.pushBack({begin, lineLen});
    }

    bool error = false;

    while(!lexer.end())
    {
        const char c = lexer.advance();
        lexer.tokenCol = lexer.col;
        lexer.tokenLine = lexer.line;
        const char* const begin = lexer.pos();

        switch(c)
        {
            case ' ':
            case '\t':
            case '\n': break;

            // single-character

            case '(': lexer.addToken(TokenType::LEFT_PAREN); break;
            case ')': lexer.addToken(TokenType::RIGHT_PAREN); break;
            case '{': lexer.addToken(TokenType::LEFT_BRACE); break;
            case '}': lexer.addToken(TokenType::RIGHT_BRACE); break;
            case ',': lexer.addToken(TokenType::COMMA); break;
            case '.': lexer.addToken(TokenType::DOT); break;
            case '-': lexer.addToken(TokenType::MINUS); break;
            case '+': lexer.addToken(TokenType::PLUS); break;
            case ';': lexer.addToken(TokenType::SEMICOLON); break;
            //case '/': // needs special case (comments)
            case '*': lexer.addToken(TokenType::STAR); break;

            // one or two character

            case '!':
            {
                lexer.addToken(lexer.match('=') ? TokenType::BANG_EQUAL : TokenType::BANG);
                break;
            }
            case '=':
            {
                lexer.addToken(lexer.match('=') ? TokenType::EQUAL_EQUAL : TokenType::EQUAL);
                break;
            }
            case '>':
            {
                lexer.addToken(lexer.match('=') ? TokenType::GREATER_EQUAL : TokenType::GREATER);
                break;
            }
            case '<':
            {
                lexer.addToken(lexer.match('=') ? TokenType::LESS_EQUAL : TokenType::LESS);
                break;
            }

            // other

            case '/':
            {
                if(!lexer.match('/'))
                    lexer.addToken(TokenType::SLASH);

                else
                {
                    while(!lexer.end() && lexer.advance() != '\n')
                        ;
                }

                break;
            }

            case '"':
            {
                const char* const literalBegin = begin + 1;

                while(!lexer.end() && lexer.advance() != '"')
                    ;

                if(lexer.end() && *lexer.pos() != '"')
                {
                    error = true;
                    printError(ErrorType::LEXER, lexer.tokenCol, lexer.tokenLine,
                            "unterminated string literal");
                }
                else
                    // "" are trimmed
                    lexer.addToken(TokenType::STRING, literalBegin, lexer.pos() - literalBegin);

                break;
            }

            default:
            {
                if(isDigit(c))
                {
                    while(!lexer.end() && isDigit(lexer.peek()))
                        lexer.advance();

                    if(lexer.peek() == '.' && isDigit(lexer.peekNext()))
                        lexer.advance();

                    while(!lexer.end() && isDigit(lexer.peek()))
                        lexer.advance();

                    const int len = lexer.pos() - begin + 1;
                    assert(getSize(_context.scratchBuf) >= len + 1);
                    memcpy(_context.scratchBuf, begin, len);
                    _context.scratchBuf[len] = '\0';
                    lexer.addToken(TokenType::NUMBER, atof(_context.scratchBuf));
                }
                else if(isAlphanumeric(c))
                {
                    while(!lexer.end() && isAlphanumeric(lexer.peek()))
                        lexer.advance();

                    const int len = lexer.pos() - begin + 1;
                    bool isKeyword = false;
                    TokenType tokenType;

                    for(const Keyword& keyword: _keywords)
                    {
                        if(int(strlen(keyword.str)) != len)
                            continue;

                        if(strncmp(begin, keyword.str, len) == 0)
                        {
                            isKeyword = true;
                            tokenType = keyword.tokenType;
                            break;
                        }
                    }

                    if(!isKeyword)
                        lexer.addToken(TokenType::IDENTIFIER, begin, lexer.pos() - begin + 1);
                    else
                        lexer.addToken(tokenType);
                }
                else
                {
                    error = true;
                    snprintf(_context.scratchBuf, getSize(_context.scratchBuf),
                            "unexpected character '%c'", c);
                    printError(ErrorType::LEXER, lexer.col, lexer.line, _context.scratchBuf);
                }

                break;
            }
        }
    }

    tokens.pushBack({TokenType::LOX_EOF, lexer.col + 1, lexer.line});
    return !error;
}

bool evaluate(Value& value, const Expr& expr);

bool isTrue(const Value& value)
{
    if(value.type == ValueType::NIL) return false;
    if(value.type == ValueType::BOOLEAN) return value.boolean;
    return true;
}

bool isEqual(const Value& l, const Value& r)
{
    if(l.type == r.type)
    {
        switch(l.type)
        {
            case ValueType::NIL: return true;
            case ValueType::BOOLEAN: return l.boolean == r.boolean;
            case ValueType::NUMBER: return l.number == r.number;
            case ValueType::STRING: return strcmp(l.string, r.string) == 0;

            default: assert(false);
        }
    }

    return false;
}

bool evaluatePrimary(Value& value, const Expr& expr)
{
    const Token& token = expr.primary.token;

    switch(token.type)
    {
        case TokenType::FALSE:
        {
            value.type = ValueType::BOOLEAN;
            value.boolean = false;
            break;
        }
        case TokenType::TRUE:
        {
            value.type = ValueType::BOOLEAN;
            value.boolean = true;
            break;
        }
        case TokenType::NIL:
        {
            value.type = ValueType::NIL;
            break;
        }
        case TokenType::NUMBER:
        {
            value.type = ValueType::NUMBER;
            value.number = token.number;
            break;
        }
        case TokenType::STRING:
        {
            value.type = ValueType::STRING;
            value.string = _context.addString(token.string.begin, token.string.len);
            break;
        }
        case TokenType::IDENTIFIER:
        {
            if(_context.newVarName.begin &&
               strncmp(token.string.begin, _context.newVarName.begin,
                        min(token.string.len, _context.newVarName.len)) == 0)
            {
                printError(ErrorType::RUNTIME, token.col, token.line,
                        "cannot read variable in its own initializer");
                return false;
            }

            const Var* var = _context.getVar(token.string.begin, token.string.len);
            if(!var)
            {
                printError(ErrorType::RUNTIME, token.col, token.line, "unknown identifier");
                return false;
            }
            value = var->value;
            break;
        }

        default: assert(false);
    }

    return true;
}

bool evaluateBinary(Value& value, const Expr& expr)
{
    const Token& token = expr.binary.operatorToken;
    Value valueLeft;
    Value valueRight;

    if(!evaluate(valueLeft, _context.expressions[expr.binary.idxExprLeft]))
        return false;

    if(token.type == TokenType::OR)
    {
        if(isTrue(valueLeft))
            value = valueLeft;
        else // evaluate right expression only if the left is false
        {
            if(!evaluate(valueRight, _context.expressions[expr.binary.idxExprRight]))
                return false;

            value = valueRight;
        }

    }
    else if(token.type == TokenType::AND)
    {
        if(!isTrue(valueLeft))
            value = valueLeft;
        else // evaluate right expression only if the left is true
        {
            if(!evaluate(valueRight, _context.expressions[expr.binary.idxExprRight]))
                return false;

            value = valueRight;
        }
    }
    else if(!evaluate(valueRight, _context.expressions[expr.binary.idxExprRight])) // @
    {
        return false;
    }
    else if(token.type == TokenType::EQUAL)
    {
        const Token& varToken = _context.expressions[expr.binary.idxExprLeft].primary.token;
        Var* var = _context.getVar(varToken.string.begin, varToken.string.len);
        assert(var); // we know it's valid - evaluate(valueLeft) returned true
        var->value = valueRight;
        value = var->value;
    }
    else if(token.type == TokenType::BANG_EQUAL)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = !isEqual(valueLeft, valueRight);
    }
    else if(token.type == TokenType::EQUAL_EQUAL)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = isEqual(valueLeft, valueRight);
    }
    else if(valueLeft.type != valueRight.type) // !!!
    {
        printError(ErrorType::RUNTIME, token.col, token.line,
                    "this operator only works with values of the same type");
        return false;
    }
    else if(token.type == TokenType::PLUS)
    {
        value.type = valueLeft.type;

        if(value.type == ValueType::NUMBER)
            value.number = valueLeft.number + valueRight.number;
        else if(value.type == ValueType::STRING)
            value.string = _context.addString(valueLeft.string, valueRight.string);
        else
        {
            printError(ErrorType::RUNTIME, token.col, token.line,
                       "operator '+' only works with strings and numbers");
            return false;
        }
    }
    else if(valueLeft.type != ValueType::NUMBER) // !!!
    {
        printError(ErrorType::RUNTIME, token.col, token.line,
                "this operator only works with numbers");
        return false;
    }
    else if(token.type == TokenType::MINUS)
    {
        value.type = ValueType::NUMBER;
        value.number = valueLeft.number - valueRight.number;
    }
    else if(token.type == TokenType::STAR)
    {
        value.type = ValueType::NUMBER;
        value.number = valueLeft.number * valueRight.number;
    }
    else if(token.type == TokenType::SLASH)
    {
        value.type = ValueType::NUMBER;
        value.number = valueLeft.number / valueRight.number;
    }
    else if(token.type == TokenType::GREATER)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = valueLeft.number > valueRight.number;
    }
    else if(token.type == TokenType::GREATER_EQUAL)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = valueLeft.number >= valueRight.number;
    }
    else if(token.type == TokenType::LESS)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = valueLeft.number < valueRight.number;
    }
    else if(token.type == TokenType::LESS_EQUAL)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = valueLeft.number <= valueRight.number;
    }
    else
        assert(false);

    return true;
}

bool evaluateUnary(Value& value, const Expr& expr)
{
    Value valueRight;
    if(!evaluate(valueRight, _context.expressions[expr.unary.idxExpr])) return false;

    const Token token = expr.unary.operatorToken;

    if(token.type == TokenType::MINUS)
    {
        if(valueRight.type != ValueType::NUMBER)
        {
            printError(ErrorType::RUNTIME, token.col, token.line,
                    "'-' unary operator works only with numbers");
            return false;
        }

        value.type = ValueType::NUMBER;
        value.number = -valueRight.number;
    }
    else // BANG
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = !isTrue(valueRight);
    }

    return true;
}

// @ GroupingExpr instead of Expr as a parameter type? (same for other expression types)
bool evaluateGrouping(Value& value, const Expr& expr)
{
    return evaluate(value, _context.expressions[expr.grouping.idxExpr]);
}

bool execute(const Stmt* const begin, const Stmt* const end);

bool evaluateCall(Value& outputValue, const Expr& expr)
{

    Value value;
    if(!evaluate(value, _context.expressions[expr.call.idxExprCallee])) return false;

    if(value.type != ValueType::CALLABLE)
    {
        printError(ErrorType::RUNTIME, expr.call.token.col, expr.call.token.line,
                "only functions can be called");
        return false;
    }

    if(expr.call.numArgs != value.callable.numParams)
    {
        snprintf(_context.scratchBuf, getSize(_context.scratchBuf),
                "expected %d arguments but got %d", value.callable.numParams, expr.call.numArgs);

        printError(ErrorType::RUNTIME, expr.call.token.col, expr.call.token.line,
                _context.scratchBuf);
        return false;
    }

    Value argValues[MAXARGS];

    for(int i = 0; i < expr.call.numArgs; ++i)
    {
        if( !evaluate(argValues[i], _context.expressions[expr.call.idxExprArg + i]) )
            return false;
    }

    if(value.callable.native)
    {
        if(!value.callable.native(outputValue, expr.call.token, argValues)) return false;
    }
    else
    {
        _context.pushEnv(true, 0); // can't be shadowed, global env is parent

        const char* paramStr = value.callable.paramStr;

        for(int i = 0; i < expr.call.numArgs; ++i)
        {
            const int paramStrLen = strlen(paramStr);
            _context.addVar(paramStr, paramStrLen, argValues[i]);
            paramStr += paramStrLen + 1; // @
        }

        _context.funReturn.success = false;

        const bool success = execute(value.callable.stmtBegin, value.callable.stmtEnd);

        if(!success && !_context.funReturn.success) // execute() really failed (runtime error)
        {
            _context.popEnv();
            return false;
        }

        if(!_context.funReturn.success)
            outputValue.type = ValueType::NIL;
        else
            outputValue = _context.funReturn.value;

        _context.popEnv();
    }

    return true;
}

bool evaluate(Value& value, const Expr& expr)
{
    switch(expr.type)
    {
        case ExprType::PRIMARY: return evaluatePrimary(value, expr);
        case ExprType::UNARY: return evaluateUnary(value, expr);
        case ExprType::GROUPING: return evaluateGrouping(value, expr);
        case ExprType::BINARY: return evaluateBinary(value, expr);
        case ExprType::CALL: return evaluateCall(value, expr);
    };
    assert(false);
}

bool print(const Expr& expr)
{
    Value value;
    if(!evaluate(value, expr)) return false;

    switch(value.type)
    {
        case ValueType::BOOLEAN: printf("%s\n", value.boolean ? "true" : "false"); break;
        case ValueType::NIL: printf("nil\n"); break;
        case ValueType::STRING: printf("%s\n", value.string); break;

        case ValueType::NUMBER:
        {
            double dummy;
            printf("%.*f\n", modf(value.number, &dummy) ? 6 : 0, value.number);
            break;
        }

        case ValueType::CALLABLE:
        {
            printf("%s\n", value.callable.native ? "native function" : "function");
            break;
        }

        default: assert(false);
    }

    return true;
}

bool declaration(Array<Stmt>& statements, const Token** const token);

bool expressionStatement(Array<Stmt>& statements, const Token** const token)
{
    Expr expr;
    if(!expression(expr, token)) return false;

    if((**token).type != TokenType::SEMICOLON)
    {
        printError(ErrorType::PARSER, (**token).col, (**token).line,
                "expected ';' after expression");
        return false;
    }

    if(expr.type == ExprType::PRIMARY) // cool, we have warnings
    {
        printError(ErrorType::PARSER, expr.primary.token.col, expr.primary.token.line,
                "warning: statement has no effect");
        // but even if has no effect don't throw it out
        // it could hide runtime errors
    }

    ++(*token);
    Stmt stmt;
    stmt.type = StmtType::EXPRESSION;
    stmt.expression.idxExpr = _context.addExpr(expr);
    statements.pushBack(stmt);
    return true;
}

bool blockStatement(Array<Stmt>& statements, const Token** const token)
{
    statements.pushBack({});
    const int idxBlock = statements.size() - 1;
    statements[idxBlock].type = StmtType::BLOCK;
    statements[idxBlock].block.canBeShadowed = true;

    while((**token).type != TokenType::RIGHT_BRACE &&
          (**token).type != TokenType::LOX_EOF)
    {
        if(!declaration(statements, token)) return false;
    }

    if((**token).type != TokenType::RIGHT_BRACE)
    {
        printError(ErrorType::PARSER, (**token).col, (**token).line,
                "expected '}' after block statement");
        return false;
    }

    ++(*token);

    statements[idxBlock].block.count = statements.size() - 1 - idxBlock;
    return true;
}

bool statement(Array<Stmt>& statements, const Token** const token)
{
    switch((**token).type)
    {
        case TokenType::PRINT:
        {
            ++(*token);

            Expr expr;
            if(!expression(expr, token)) return false;

            if((**token).type != TokenType::SEMICOLON)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected ';' after expression");
                return false;
            }

            ++(*token);
            Stmt stmt;
            stmt.type = StmtType::PRINT;
            stmt.print.idxExpr = _context.addExpr(expr);
            statements.pushBack(stmt);
            break;
        }

        case TokenType::LEFT_BRACE:
        {
            ++(*token);
            return blockStatement(statements, token);
        }

        case TokenType::IF:
        {
            ++(*token);

            if((**token).type != TokenType::LEFT_PAREN)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected '(' after if");
                return false;
            }

            ++(*token);

            statements.pushBack({});

            // we can't use pointers here due to pointer invalidation

            const int idxIf = statements.size() - 1;
            statements[idxIf].type = StmtType::IF_STMT;

            {
                Expr expr;
                if(!expression(expr, token)) return false;
                statements[idxIf].ifStmt.idxExpr = _context.addExpr(expr);
            }

            if((**token).type != TokenType::RIGHT_PAREN)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected ')' after if condition");
                return false;
            }

            ++(*token);

            if(!statement(statements, token)) return false;

            statements[idxIf].ifStmt.count = statements.size() - 1 - idxIf;

            if((**token).type != TokenType::ELSE)
                statements[idxIf].ifStmt.elseCount = 0;
            else
            {
                ++(*token);
                if(!statement(statements, token)) return false;

                statements[idxIf].ifStmt.elseCount = statements.size() - 1 -
                    (idxIf + statements[idxIf].ifStmt.count);
            }

            break;
        }

        case TokenType::WHILE:
        {
            ++(*token);

            if((**token).type != TokenType::LEFT_PAREN)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected '(' after while");
                return false;
            }

            ++(*token);

            statements.pushBack({});
            const int idxWhile = statements.size() - 1;
            statements[idxWhile].type = StmtType::WHILE_STMT;

            {
                Expr expr;
                if(!expression(expr, token)) return false;
                statements[idxWhile].whileStmt.idxExpr = _context.addExpr(expr);
            }

            if((**token).type != TokenType::RIGHT_PAREN)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected ')' after while condition");
                return false;
            }

            ++(*token);
            if(!statement(statements, token)) return false;
            statements[idxWhile].whileStmt.count = statements.size() - 1 - idxWhile;
            break;
        }

        case TokenType::FOR:
        {
            ++(*token);

            if((**token).type != TokenType::LEFT_PAREN)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected '(' after for");
                return false;
            }

            ++(*token);

            // so we won't 'leak' a variable from init statement
            statements.pushBack({});
            const int idxBlock = statements.size() - 1;
            statements[idxBlock].type = StmtType::BLOCK;
            statements[idxBlock].block.canBeShadowed = false;

            // 1. init statement
            if((**token).type == TokenType::SEMICOLON)
            {
                ++(*token);
            }
            else if((**token).type == TokenType::VAR)
            {
                if(!declaration(statements, token)) return false;
            }
            else
            {
                if(!expressionStatement(statements, token)) return false;
            }

            // 2. condition
            int idxConditionExpr = -1;
            if((**token).type != TokenType::SEMICOLON)
            {
                Expr expr;
                if(!expression(expr, token)) return false;
                idxConditionExpr = _context.addExpr(expr);
            }

            if((**token).type != TokenType::SEMICOLON)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected ';' after loop condition");
                return false;
            }

            ++(*token);

            // 3. increment
            int idxIncrementExpr = -1;
            if((**token).type != TokenType::RIGHT_PAREN)
            {
                Expr expr;
                if(!expression(expr, token)) return false;
                idxIncrementExpr = _context.addExpr(expr);
            }

            if((**token).type != TokenType::RIGHT_PAREN)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected ')' after for clauses");
                return false;
            }

            ++(*token);

            // implement in terms of a while statement
            statements.pushBack({});
            const int idxWhile = statements.size() - 1;
            statements[idxWhile].type = StmtType::WHILE_STMT;

            if(idxConditionExpr != -1)
            {
                statements[idxWhile].whileStmt.idxExpr = idxConditionExpr;
            }
            else
            {
                Expr expr;
                expr.type = ExprType::PRIMARY;
                expr.primary.token = {TokenType::TRUE};
                statements[idxWhile].whileStmt.idxExpr = _context.addExpr(expr);
            }

            if(!statement(statements, token)) return false;

            if(idxIncrementExpr != -1)
            {
                Stmt stmt;
                stmt.type = StmtType::EXPRESSION;
                stmt.expression.idxExpr = idxIncrementExpr;
                statements.pushBack(stmt);
            }

            statements[idxWhile].whileStmt.count = statements.size() - 1 - idxWhile;
            statements[idxBlock].block.count = statements.size() - 1 - idxBlock;
            break;
        }

        case TokenType::RETURN:
        {
            ++(*token);

            Expr expr;
            expr.type = ExprType::PRIMARY;
            expr.primary.token = {TokenType::NIL};

            if((**token).type != TokenType::SEMICOLON)
            {
                if(!expression(expr, token)) return false;
            }

            if((**token).type != TokenType::SEMICOLON)
            {
                printError(ErrorType::PARSER, (**token).col, (**token).line,
                        "expected ';' after return statement");
                return false;
            }

            ++(*token);

            Stmt stmt;
            stmt.type = StmtType::RETURN_STMT;
            stmt.returnStmt.idxExpr = _context.addExpr(expr);
            statements.pushBack(stmt);
            break;
        }

        default:
            return expressionStatement(statements, token);
    }

    return true;
}

// we keep declaration() and statement() separately because sometimes we
// want a statement that is not a variable or function declaration

bool declaration(Array<Stmt>& statements, const Token** const token)
{
    if((**token).type == TokenType::VAR)
    {
        ++(*token);

        // @ we should have some helper functions for these
        if((**token).type != TokenType::IDENTIFIER)
        {
            printError(ErrorType::PARSER, (**token).col, (**token).line,
                    "expected variable name ");
            return false;
        }

        Stmt stmt;
        stmt.varDecl.identifierToken = **token;
        ++(*token);

        if((**token).type == TokenType::EQUAL)
        {
            ++(*token);

            Expr expr;
            if(!expression(expr, token)) return false;
            stmt.varDecl.idxExpr = _context.addExpr(expr);
        }
        else // init to nil
        {
            Expr expr;
            expr.type = ExprType::PRIMARY;
            expr.primary.token.type = TokenType::NIL;
            stmt.varDecl.idxExpr = _context.addExpr(expr);
        }

        if((**token).type != TokenType::SEMICOLON)
        {
            printError(ErrorType::PARSER, (**token).col, (**token).line, "expected ';'");
            return false;
        }

        ++(*token);
        stmt.type = StmtType::VAR_DECL;
        statements.pushBack(stmt);
        return true;
    }
    else if((**token).type == TokenType::FUN)
    {
        ++(*token);

        if((**token).type != TokenType::IDENTIFIER)
        {
            printError(ErrorType::PARSER, (**token).col, (**token).line,
                    "expected function name");
            return false;
        }

        statements.pushBack({});
        const int idxFunDecl = statements.size() - 1;
        statements[idxFunDecl].type = StmtType::FUN_DECL;
        statements[idxFunDecl].funDecl.identifierToken = **token;
        statements[idxFunDecl].funDecl.numParams = 0;

        ++(*token);

        if((**token).type != TokenType::LEFT_PAREN)
        {
            printError(ErrorType::PARSER, (**token).col, (**token).line,
                    "expected '(' after function name");
            return false;
        }

        ++(*token);

        if((**token).type != TokenType::RIGHT_PAREN)
        {
            while(true)
            {
                if(statements[idxFunDecl].funDecl.numParams == MAXARGS)
                {
                    snprintf(_context.scratchBuf, getSize(_context.scratchBuf),
                            "LOX supports only %d function arguments", MAXARGS);

                    printError(ErrorType::PARSER, (**token).col, (**token).line,
                            _context.scratchBuf);
                    return false;
                }

                if((**token).type != TokenType::IDENTIFIER)
                {
                    printError(ErrorType::PARSER, (**token).col, (**token).line,
                            "expected parameter name");
                    return false;
                }

                const char* paramStr = _context.addString((**token).string.begin,
                        (**token).string.len);

                if(statements[idxFunDecl].funDecl.numParams == 0)
                    statements[idxFunDecl].funDecl.paramStr = paramStr;

                statements[idxFunDecl].funDecl.numParams += 1;
                ++(*token);

                if((**token).type != TokenType::COMMA)
                    break;

                ++(*token);
            }
        }

        if((**token).type != TokenType::RIGHT_PAREN)
        {
            printError(ErrorType::PARSER, (**token).col, (**token).line,
                    "expected ')' after function parameters");
            return false;
        }

        ++(*token);

        if((**token).type != TokenType::LEFT_BRACE)
        {

            printError(ErrorType::PARSER, (**token).col, (**token).line,
                    "expected '{' before function body");
            return false;
        }

        ++(*token);

        if(!blockStatement(statements, token)) return false;
        statements[idxFunDecl].funDecl.count = statements.size() - 1 - idxFunDecl;
        return true;
    }
    else
        return statement(statements, token);
}

bool parse(Array<Stmt>& statements, const Array<Token>& tokens)
{
    bool error = false;
    const Token* token = tokens.begin();

    // continue parsing even if a statement is incorrect so we can report many syntax errors
    // at once
    while(token->type != TokenType::LOX_EOF)
    {
        if(!declaration(statements, &token))
        {
            error = true;

            // go to the next statement
            // it is not 100% accurate but that's fine
            // this should be inside declaration() for better results

            while(token->type != TokenType::LOX_EOF)
            {
                ++token;

                if((token - 1)->type == TokenType::SEMICOLON)
                    break;

                switch(token->type)
                {
                    case TokenType::CLASS:
                    case TokenType::FUN:
                    case TokenType::VAR:
                    case TokenType::FOR:
                    case TokenType::IF:
                    case TokenType::WHILE:
                    case TokenType::PRINT:
                    case TokenType::RETURN: goto end;
                }
            }
            end:;
        }
    }

    return !error;
}

bool execute(const Stmt* const begin, const Stmt* const end)
{
    for(const Stmt* stmt = begin; stmt != end; ++stmt)
    {
        assert(stmt < end);

        switch(stmt->type)
        {
            case StmtType::EXPRESSION:
            {
                Value value;
                if(!evaluate(value, _context.expressions[stmt->expression.idxExpr])) return false;
                break;
            }

            case StmtType::PRINT:
            {
                if(!print(_context.expressions[stmt->print.idxExpr])) return false;
                break;
            }

            case StmtType::VAR_DECL:
            {
                const Token& token = stmt->varDecl.identifierToken;

                _context.newVarName.begin = token.string.begin;
                _context.newVarName.len = token.string.len;

                Value value;
                if(!evaluate(value, _context.expressions[stmt->varDecl.idxExpr])) return false;

                if(!_context.addVar(token.string.begin, token.string.len, value))
                {
                    printError(ErrorType::RUNTIME, token.col, token.line,
                            "only global variables can be redefined");
                    return false;
                }

                _context.newVarName.begin = nullptr;

                break;
            }

            case StmtType::FUN_DECL:
            {
                Value value;
                value.type = ValueType::CALLABLE;
                value.callable.numParams = stmt->funDecl.numParams;
                value.callable.native = nullptr;
                value.callable.stmtBegin = stmt + 1;
                value.callable.stmtEnd = value.callable.stmtBegin + stmt->funDecl.count;

                value.callable.paramStr = stmt->funDecl.paramStr;

                const Token& token = stmt->funDecl.identifierToken;

                if(!_context.addVar(token.string.begin, token.string.len, value))
                {
                    printError(ErrorType::RUNTIME, token.col, token.line,
                            "only global functions can be redefined");
                    return false;
                }

                // we don't want to execute the function, it is only a declaration...
                stmt += stmt->funDecl.count;
                break;
            }

            case StmtType::BLOCK:
            {
                _context.pushEnv(stmt->block.canBeShadowed);

                if(!execute(stmt + 1, stmt + 1 + stmt->block.count))
                {
                    _context.popEnv();
                    return false;
                }

                _context.popEnv();
                stmt += stmt->block.count;
                break;
            }

            case StmtType::IF_STMT:
            {
                Value value;
                const Stmt* lastChildStmt = stmt + stmt->ifStmt.count + stmt->ifStmt.elseCount;

                if(!evaluate(value, _context.expressions[stmt->ifStmt.idxExpr])) return false;

                if(isTrue(value))
                {
                    if(!execute(stmt + 1, stmt + 1 + stmt->ifStmt.count)) return false;
                }
                else if(stmt->ifStmt.elseCount)
                {
                    if(!execute(stmt + stmt->ifStmt.count + 1, lastChildStmt + 1)) return false;
                }

                stmt = lastChildStmt;
                break;
            }

            case StmtType::WHILE_STMT:
            {
                const Stmt* lastBodyStmt = stmt + stmt->whileStmt.count;

                for(;;)
                {
                    Value value;

                    if(!evaluate(value, _context.expressions[stmt->whileStmt.idxExpr]))
                        return false;

                    if(!isTrue(value))
                        break;

                    if(!execute(stmt + 1, lastBodyStmt + 1))
                        return false;
                }

                stmt = lastBodyStmt;
                break;
            }

            case StmtType::RETURN_STMT:
            {
                Value value;

                if(!evaluate(value, _context.expressions[stmt->returnStmt.idxExpr]))
                    return false;

                _context.funReturn.success = true;
                _context.funReturn.value = value;

                return false; // hacky...
            }

            default: assert(false);
        }
    }
    return true;
}

void trimEndWhitespace(Array<char>& source)
{
    assert(source.back() == '\0');

    // omit '\0'
    for(char* it = source.end() - 2; it >= source.begin(); --it)
    {
        if(*it == ' ' || *it == '\n' || *it == '\t')
            *it = '\0';
        else
            break;
    }
}

void keyboardInterrupt(int)
{
    // @
    printf("\nuse Ctrl-D (i.e. EOF) to exit\n>>> ");
    fflush(stdout);
}

int main(int argc, const char* const * const argv)
{
    if(argc > 2)
    {
        printf("usage:\nslox\nslox <source>\n");
        return 0;
    }

    Array<char> source;
    const bool repl = (argc == 1) && isatty(fileno(stdin));

    if(argc == 2) // file input
    {
        FILE* file = fopen(argv[1], "r");
        if(!file)
        {
            printf("error: could not open '%s'\n", argv[1]);
            return 0;
        }

        if(fseek(file, 0, SEEK_END) != 0)
        {
            perror("fseek()");
            fclose(file);
            return 0;
        }

        const int size = ftell(file);

        if(size == EOF)
        {
            perror("ftell()");
            fclose(file);
            return 0;
        }

        rewind(file);
        source.resize(size);
        // might read less than size on windows - e.g. \r\n to \n conversion
        // to match ftell() we have to open the file in a binary mode
        fread(source.data(), 1, size, file);
        fclose(file);
        source.pushBack('\0');
        trimEndWhitespace(source);
    }
    else if(!repl) // pipe input
    {
        // it is probably not the fastest way to get the data from stdin
        source.reserve(500);

        char c;
        while( (c = getchar()) != EOF)
            source.pushBack(c);

        source.pushBack('\0');
        trimEndWhitespace(source);
    }
    else // repl session
    {
        source.reserve(500);
        // but now we can't terminate when a script is running
        signal(SIGINT, keyboardInterrupt);
        printf("super LOX - implementation of craftinginterpreters.com LOX language in C++\n");
    }

    Array<Token> tokens;
    Array<Stmt> statements;

    // allocate some memory upfront
    tokens.reserve(1000);
    statements.reserve(100);
    _context.lines.reserve(100);
    _context.expressions.reserve(1000);
    _context.variables.reserve(30);
    _context.environments.reserve(10);

    for(;;)
    {
        if(repl)
        {
            printf(">>> ");

            // @TODO:
            // we shouldn't clear these things; and we shouldn't grow them, wo we don't
            // invalidate the pointers...
            // it works if functions are not used...
            // need to change the design
            statements.clear();
            source.clear();
            tokens.clear();
            _context.lines.clear();
            _context.expressions.clear();

            // don't call this; this will free all the strings that our runtimes uses
            // for e.g. variable names

            // _context.end = 0;

            for(;;)
            {
                const int c = getchar();

                if(c == '\n')
                {
                    source.pushBack('\0');
                    trimEndWhitespace(source);
                    break;
                }
                else if(c == EOF)
                {
                    if(source.empty())
                    {
                        printf("\nbye\n");
                        return 0;
                    }
                    else // ignore EOF
                        clearerr(stdin);
                }
                else
                    source.pushBack(c);
            }
        }

        bool error = false;
        error = !scan(tokens, source.data()) || error;
        error = !parse(statements, tokens) || error;

        if(!error)
            execute(statements.begin(), statements.end());

        if(!repl)
            break;
    }

    return 0;
}
