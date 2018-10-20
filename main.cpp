#include "Array.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h> // check if stdin is terminal or pipe
#include <signal.h> // keyboard interrupt

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
    {"this", TokenType::THIS}, {"ver", TokenType::VAR}, {"while", TokenType::WHILE}
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
    NIL
};

struct Value
{
    ValueType type;

    union
    {
        bool boolean;
        double number;
        const char* string;
    };
};

enum class ExprType
{
    BINARY,
    GROUPING,
    LITERAL,
    UNARY
};

struct Expr
{
    ExprType type; // we evaluate the expression based on its type

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
        } grouping; // @ we need 'grouping' for some reason; I don't know why yet

        struct
        {
            int idxExpr;
            Token operatorToken;
        } unary;

        Token literalToken;
    };
};

struct Line
{
    const char* begin;
    int len;
};

struct
{
    Array<Line> lines;
    Array<Expr> expressions;
    char scratchBuf[10000];

    // temp storage for string values
    char byteArray[10000];
    int end = 0;

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

enum class ErrorType
{
    LEXER,
    PARSER,
    RUNTIME
};

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
    }

    printf("^\n%d:%d: {%s} %s\n\n", line, col, typeStr, str);
}

bool expression(Expr& expr, const Token** const token);

bool primary(Expr& outputExpr, const Token** const token)
{
    const TokenType type = (**token).type;

    if(type == TokenType::FALSE || type == TokenType::TRUE || type == TokenType::NIL ||
       type == TokenType::NUMBER || type == TokenType::STRING)
    {
        outputExpr.type = ExprType::LITERAL;
        outputExpr.literalToken = **token;
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

    return primary(outputExpr, token);
}

enum class BinaryType
{
    EQUALITY,
    COMPARISON,
    ADDITION,
    MULTIPLICATION
};

// all LOX binary operators are left-associative

bool binary(Expr& expr, const Token** const token, BinaryType binaryType)
{
    TokenType tokenTypes[5] = {};
    BinaryType nextBinaryType;

    switch(binaryType)
    {
        case BinaryType::EQUALITY:
        {
            nextBinaryType = BinaryType::COMPARISON;
            tokenTypes[0] = TokenType::EQUAL_EQUAL;
            tokenTypes[1] = TokenType::BANG_EQUAL;
            break;
        }
        case BinaryType::COMPARISON:
        {
            nextBinaryType = BinaryType::ADDITION;
            tokenTypes[0] = TokenType::GREATER;
            tokenTypes[1] = TokenType::GREATER_EQUAL;
            tokenTypes[2] = TokenType::LESS;
            tokenTypes[3] = TokenType::LESS_EQUAL;
            break;
        }
        case BinaryType::ADDITION:
        {
            nextBinaryType = BinaryType::MULTIPLICATION;
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

    if(binaryType != BinaryType::MULTIPLICATION)
    { if(!binary(expr, token, nextBinaryType)) return false; }
    else
    { if(!unary(expr, token)) return false;}

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

        Token operatorToken = **token;
        ++(*token);

        Expr exprRight;

        if(binaryType != BinaryType::MULTIPLICATION)
        { if(!binary(exprRight, token, nextBinaryType)) return false; }
        else
        { if(!unary(exprRight, token)) return false;}

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
    return binary(expr, token, BinaryType::EQUALITY);
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

    const char* pos() const {return it - 1;} // @ in fact it is a previous position
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
                        if(strncmp(begin, keyword.str, min(len, int(strlen(keyword.str)))) == 0)
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

bool evaluateLiteral(Value& value, const Expr& expr)
{
    const Token& token = expr.literalToken;

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

        default: assert(false);
    }

    return true;
}

bool evaluateBinary(Value& value, const Expr& expr)
{
    Value valueLeft;
    Value valueRight;

    if(!evaluate(valueLeft, _context.expressions[expr.binary.idxExprLeft])) return false;
    if(!evaluate(valueRight, _context.expressions[expr.binary.idxExprRight])) return false;

    const Token& token = expr.binary.operatorToken;

    if(token.type == TokenType::BANG_EQUAL)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = !isEqual(valueLeft, valueRight);
    }
    else if(token.type == TokenType::EQUAL_EQUAL)
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = isEqual(valueLeft, valueRight);
    }
    else
    {
        if(valueLeft.type != valueRight.type)
        {
            printError(ErrorType::RUNTIME, token.col, token.line,
                        "this operator only works with values of the same type");
            return false;
        }

        if(token.type == TokenType::PLUS)
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
        else
        {
            if(valueLeft.type != ValueType::NUMBER)
            {
                printError(ErrorType::RUNTIME, token.col, token.line,
                        "this operator only works with numbers");
                return false;
            }

            if(token.type == TokenType::MINUS)
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
        }
    }

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
    // BANG
    else
    {
        value.type = ValueType::BOOLEAN;
        value.boolean = !isTrue(valueRight);
    }

    return true;
}

// @ GroupingExpr instead of Expr as parameter type?
bool evaluateGrouping(Value& value, const Expr& expr)
{
    return evaluate(value, _context.expressions[expr.grouping.idxExpr]);
}

bool evaluate(Value& value, const Expr& expr)
{
    switch(expr.type)
    {
        case ExprType::LITERAL: return evaluateLiteral(value, expr);
        case ExprType::UNARY: return evaluateUnary(value, expr);
        case ExprType::GROUPING: return evaluateGrouping(value, expr);
        case ExprType::BINARY: return evaluateBinary(value, expr);
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
        case ValueType::STRING: printf("\"%s\"\n", value.string); break;
        case ValueType::NUMBER: printf("%f\n", value.number); break;

        default: assert(false);
    }

    return true;
}

enum class StmtType
{
    EXPRESSION,
    PRINT
};

struct Stmt
{
    StmtType type;
    int idxExpr;
};

bool statement(Stmt& stmt, const Token** const token)
{
    if((**token).type == TokenType::PRINT)
    {
        ++(*token);
        stmt.type = StmtType::PRINT;
    }
    else
        stmt.type = StmtType::EXPRESSION;

    Expr expr;
    if(!expression(expr, token)) return false;

    if((**token).type != TokenType::SEMICOLON)
    {
        printError(ErrorType::PARSER, (**token).col, (**token).line, "expected ';'");
        return false;
    }

    ++(*token);

    stmt.idxExpr = _context.addExpr(expr);

    return true;
}

bool parse(Array<Stmt>& statements, const Array<Token>& tokens)
{
    bool error = false;

    const Token* token = tokens.begin();

    while(token->type != TokenType::LOX_EOF)
    {
        statements.pushBack({});

        if(!statement(statements.back(), &token))
        {
            error = true;

            // go to the next statement and continue parsing (it's not always correct)
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

void execute(Array<Stmt>& statements)
{
    for(Stmt& stmt: statements)
    {
        if(stmt.type == StmtType::EXPRESSION)
        {
            Value value;

            if(!evaluate(value, _context.expressions[stmt.idxExpr]))
                break;
        }
        else
        {
            if(!print(_context.expressions[stmt.idxExpr]))
                break;
        }
    }
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
        // to match ftell() we have to open the file in binary mode
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
        signal(SIGINT, keyboardInterrupt);
        printf("super LOX - implementation of craftinginterpreters.com LOX language\n");
    }

    Array<Token> tokens;
    Array<Stmt> statements;

    // allocate some memory upfront
    tokens.reserve(1000);
    statements.reserve(100);
    _context.lines.reserve(100);
    _context.expressions.reserve(1000);

    for(;;)
    {
        if(repl)
        {
            printf(">>> ");

            source.clear();
            tokens.clear();
            statements.clear();
            _context.lines.clear();
            _context.expressions.clear();
            _context.end = 0;

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
            execute(statements);

        if(!repl)
            break;
    }

    return 0;
}
