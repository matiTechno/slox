#include "Array.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
    BOOL,
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

    int addExpr(const Expr expr)
    {
        expressions.pushBack(expr);
        return expressions.size() - 1;
    }

} static _context;

void printError(int col, int line, const char* str)
{
    printf("%.*s\n", _context.lines[line - 1].len, _context.lines[line - 1].begin);

    for(int i = 0; i < col - 1; ++i)
        putchar('-');

    printf("^\n%d:%d: %s\n", line, col, str);
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
            printError((**token).col, (**token).line, "expected ')' after expression");
            return false;
        }
    }

    printError((**token).col, (**token).line, "expected expression");
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

        expr.binary.idxExprLeft = _context.addExpr(expr); // add expr to vector before modifying it
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

bool parse(Expr& expr, const Array<Token>& tokens)
{
    const Token* token = tokens.begin();
    return expression(expr, &token);
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

    const char* pos() const {return it - 1;}
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
    _context.lines.clear();
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

                if(lexer.end())
                {
                    error = true;
                    printError(lexer.tokenCol, lexer.tokenLine, "unterminated string literal");
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

                    const int len = min(int(lexer.pos() - begin + 1),
                                        int(sizeof(_context.scratchBuf - 1)));

                    memcpy(_context.scratchBuf, begin, len);
                    _context.scratchBuf[len + 1] = '\0';
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
                    printError(lexer.col, lexer.line, _context.scratchBuf);
                }

                break;
            }
        }
    }

    tokens.pushBack({TokenType::LOX_EOF, lexer.col + 1, lexer.line});
    return !error;
}

bool evaluate(Value& value, const Expr& expr)
{
    (void)value;
    (void)expr;
    return false;
}

void print(const Value& value)
{
    (void)value;
}

int main()
{
    // @ remove extra whitespace from the end (better error reporting)
    const char* const source = "2.666 * 3 + (1 + 8) - 5 / (2 + --6.5)";

    Array<Token> tokens;
    Expr expr;
    bool error = false;

    error = scan(tokens, source) || error;
    error = parse(expr, tokens) || error;

    if(!error)
    {
        Value value;

        if(evaluate(value, expr))
            print(value);
    }

    return 0;
}
