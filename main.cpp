#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Array.hpp"

bool isDigit(const char c) {return c >= '0' && c <= '9';}

bool isAlphanumeric(const char c)
{
    return isDigit(c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

struct Token
{
    enum Type
    {
        // single character tokens
        LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE,
        COMMA, DOT, MINUS, PLUS, SEMICOLON, SLASH, STAR,

        // one or two character
        BANG, BANG_EQUAL,
        EQUAL, EQUAL_EQUAL,
        GREATER, GREATER_EQUAL,
        LESS, LESS_EQUAL,

        // literls
        IDENTIFIER, STRING, NUMBER,

        // keywords
        AND, CLASS, ELSE, FALSE, FUN, FOR, IF, NIL, OR,
        PRINT, RETURN, SUPER, THIS, TRUE, VAR, WHILE,

        LOX_EOF,

        NOT_INIT
    };

    Type type;
    int line;

    union
    {
        struct
        {
            const char* ptr;
            int len;
        } str;

        double number;
    };
};

struct Keyword
{
    const char* str;
    Token::Type tokenType;
};

static const Keyword _keywords[] =
{
    {"and", Token::AND}, {"class", Token::CLASS}, {"else", Token::ELSE}, {"false", Token::FALSE},
    {"fun", Token::FUN}, {"for", Token::FOR}, {"if", Token::IF}, {"nil", Token::NIL},
    {"or", Token::OR}, {"print", Token::PRINT}, {"return", Token::RETURN}, {"super", Token::SUPER},
    {"this", Token::THIS}, {"ver", Token::VAR}, {"while", Token::WHILE}
};

struct Expr
{
    Expr* left = nullptr;
    Token token = {Token::NOT_INIT};
    Expr* right = nullptr;
};

Token evaluate(const Expr& expr)
{
    Token tokLeft, tokRight;

    if(expr.left)
        tokLeft = evaluate(*expr.left);

    if(expr.right)
        tokRight = evaluate(*expr.right);

    switch(expr.token.type)
    {
        case Token::NUMBER:
        case Token::STRING: return expr.token;

        case Token::PLUS:
        {
            Token tok;
            tok.type = Token::NUMBER;
            // @ report error if tokLeft or tokRight are not numbers,
            // or if tokLeft or toRight are uninitialized
            tok.number = tokLeft.number + tokRight.number;
            return tok;
        }
        case Token::MINUS:
        {
            Token tok;
            tok.type = Token::NUMBER;

            if(tokLeft.type == Token::NOT_INIT)
                tok.number = -tokRight.number;
            else
                tok.number = tokLeft.number - tokRight.number;

            return tok;
        }
        case Token::STAR:
        {
            Token tok;
            tok.type = Token::NUMBER;
            tok.number = tokLeft.number * tokRight.number;
            return tok;
        }
        case Token::SLASH:
        {
            Token tok;
            tok.type = Token::NUMBER;
            tok.number = tokLeft.number / tokRight.number;
            return tok;
        }

        // @
        default: return {Token::NIL};
    }
}

void interpret(const Expr& expr)
{
    Token token = evaluate(expr);

    printf("> ");

    switch(token.type)
    {
        case Token::NUMBER: printf("%f\n", token.number); break;
        case Token::STRING: printf("%.*s\n", token.str.len, token.str.ptr); break;

        default: break;
    }
}

template<typename T, int N>
int getSize(T(&)[N]) {return N;}

struct
{
    Expr data[10000];
    int endIdx = 0;

    Expr* push(const Expr& expr)
    {
        assert(endIdx < getSize(data) - 1);
        data[endIdx] = expr;
        ++endIdx;
        return &data[endIdx - 1];
    }

} static exprArray;

#define CHECK(x) x if(error) return {};

struct Parser
{
    Parser(const Array<Token>& tokens): tokens(tokens) {}

    const Array<Token>& tokens;
    int tokIdx;
    bool error;

    bool parse(Expr& expr)
    {
        tokIdx = 0;
        exprArray.endIdx = 0;
        error = false;

        expr = expression();

        return !error;
    }

    Expr expression()
    {
        return equality();
    }

    Expr equality()
    {
        CHECK( Expr expr = comparison(); )

        while( (tokIdx < tokens.size()) &&
               (tokens[tokIdx].type == Token::BANG_EQUAL ||
                    tokens[tokIdx].type == Token::EQUAL_EQUAL) )
        {
            Expr* exprLeft = exprArray.push(expr);

            Token op = tokens[tokIdx];
            ++tokIdx;

            CHECK( Expr* exprRight = exprArray.push(comparison()); )

            expr = {exprLeft, op, exprRight};
        }

        return expr;
    }

    Expr comparison()
    {
        CHECK( Expr expr = addition(); )

        while( (tokIdx < tokens.size()) &&
               (tokens[tokIdx].type == Token::GREATER ||
                    tokens[tokIdx].type == Token::GREATER_EQUAL ||
                    tokens[tokIdx].type == Token::LESS ||
                    tokens[tokIdx].type == Token::LESS_EQUAL) )
        {
            Expr* exprLeft = exprArray.push(expr);

            Token op = tokens[tokIdx];
            ++tokIdx;

            CHECK( Expr* exprRight = exprArray.push(addition()); )

            expr = {exprLeft, op, exprRight};
        }

        return expr;
    }

    Expr addition()
    {
        CHECK( Expr expr = multiplication(); )

        while( (tokIdx < tokens.size()) &&
               (tokens[tokIdx].type == Token::MINUS || tokens[tokIdx].type == Token::PLUS) )
        {
            Expr* exprLeft = exprArray.push(expr);

            Token op = tokens[tokIdx];
            ++tokIdx;

            CHECK( Expr* exprRight = exprArray.push(multiplication()); )

            expr = {exprLeft, op, exprRight};
        }

        return expr;
    }

    Expr multiplication()
    {
        CHECK( Expr expr = unary(); )

        while( (tokIdx < tokens.size()) &&
               (tokens[tokIdx].type == Token::SLASH || tokens[tokIdx].type == Token::STAR) )
        {
            Expr* exprLeft = exprArray.push(expr);

            Token op = tokens[tokIdx];
            ++tokIdx;

            CHECK( Expr* exprRight = exprArray.push(unary()); )

            expr = {exprLeft, op, exprRight};
        }

        return expr;
    }

    Expr unary()
    {
        if(error)
            return {};

        if( (tokIdx < tokens.size()) &&
            (tokens[tokIdx].type == Token::BANG || tokens[tokIdx].type == Token::MINUS) )
        {
            Token op = tokens[tokIdx];
            ++tokIdx;
            
            Expr* exprRight = exprArray.push(unary());
            return {nullptr, op, exprRight};
        }

        return primary();
    }

    Expr primary()
    {
        if( (tokIdx < tokens.size()) &&
            (tokens[tokIdx].type == Token::FALSE || tokens[tokIdx].type == Token::TRUE ||
                tokens[tokIdx].type == Token::NIL || tokens[tokIdx].type == Token::NUMBER ||
                tokens[tokIdx].type == Token::STRING) )
        {
            Expr expr = {nullptr, tokens[tokIdx], nullptr};
            ++tokIdx;
            return expr;
        }
        else if( (tokIdx < tokens.size()) && (tokens[tokIdx].type == Token::LEFT_PAREN) )
        {
            ++tokIdx;
            CHECK( const Expr expr = expression(); )

            if( (tokIdx < tokens.size()) && (tokens[tokIdx].type == Token::RIGHT_PAREN) )
            {
                ++tokIdx;
                return expr;
            }
            else
            {
                printf("%d: expected ')' after expression\n", tokens[tokIdx].line);
                error = true;
                return {};
            }
        }

        printf("%d: expect expression\n", tokens[tokIdx].line);
        error = true;
        return {};
    }
};

bool scan(Array<Token>& tokens, const char* const str)
{
    bool error = false;
    const int len = strlen(str);
    int idx = 0;
    int line = 1;

    while(idx < len)
    {
        const char c = str[idx];
        const char cnext = str[idx + 1];

        switch(c)
        {
            case '(': tokens.pushBack({Token::LEFT_PAREN, line }); break;
            case ')': tokens.pushBack({Token::RIGHT_PAREN, line }); break;
            case '{': tokens.pushBack({Token::LEFT_BRACE, line}); break;
            case '}': tokens.pushBack({Token::RIGHT_BRACE, line}); break;
            case ',': tokens.pushBack({Token::COMMA, line}); break;
            case '.': tokens.pushBack({Token::DOT, line}); break;
            case '-': tokens.pushBack({Token::MINUS, line}); break;
            case '+': tokens.pushBack({Token::PLUS, line}); break;
            case ';': tokens.pushBack({Token::SEMICOLON, line}); break;
            //case '/': tokens.pushBack({Token::SLASH}); break; // needs special case (comments)
            case '*': tokens.pushBack({Token::STAR, line}); break;

            case '!': tokens.pushBack({cnext == '=' ? (++idx, Token::BANG_EQUAL) : Token::BANG,
                              line});
                      break;

            case '=': tokens.pushBack({cnext == '=' ? (++idx, Token::EQUAL_EQUAL) : Token::EQUAL,
                              line});
                      break;
            case '>': tokens.pushBack({cnext == '=' ? (++idx, Token::GREATER_EQUAL) :
                              Token::GREATER, line});
                      break;

            case '<': tokens.pushBack({cnext == '=' ? (++idx, Token::LESS_EQUAL) : Token::LESS,
                              line});
                      break;

            case '/':
            {
                if(cnext != '/')
                    tokens.pushBack({Token::SLASH, line});

                else
                {
                    while(++idx, str[idx] != '\0' && str[idx] != '\n')
                        ;

                    ++line;
                }

                break;
            }

            case ' ':
            case '\t':
            break;

            case '\n': ++line; break;

            case '"':
            {
                const int begin = idx + 1;
                const int beginLine = line;

                while(++idx, str[idx] != '\0' && str[idx] != '"')
                {
                    if(str[idx] == '\n')
                        ++line;
                }


                if(str[idx] != '"')
                {
                    error = true;
                    printf("%d: unterminated string literal\n", beginLine);
                    break;
                }
                else
                {
                    Token token;
                    token.type = Token::STRING;
                    token.line = beginLine;
                    token.str.ptr = &str[begin];
                    token.str.len = idx - begin;
                    tokens.pushBack(token);
                    break;
                }
            }

            default:
            {
                Token token;
                token.line = line;

                if(isDigit(c))
                {
                    const int begin = idx;

                    while(++idx, isDigit(str[idx]))
                        ;

                    if( str[idx] == '.' && isDigit(str[idx + 1]) )
                        ++idx;

                    while(isDigit(str[idx]))
                        ++idx;

                    char buf[256];
                    const int len = idx - begin;
                    assert( len < int(sizeof(buf)) );

                    memcpy(buf, &str[begin], len);
                    buf[len + 1] = '\0';

                    token.number = atof(buf);
                    token.type = Token::NUMBER;

                    --idx;
                }
                else if(isAlphanumeric(c))
                {
                    const int begin = idx;

                    while(++idx, isAlphanumeric(str[idx]))
                        ;

                    const int len = idx - begin;
                    bool match = false;

                    for(const Keyword& keyword: _keywords)
                    {
                        const int keywordLen = strlen(keyword.str);
                        
                        if(strncmp(&str[begin], keyword.str, len > keywordLen ? keywordLen: len)
                                == 0)
                        {
                            token.type = keyword.tokenType;
                            match = true;
                            break;
                        }
                    }

                    if(!match)
                    {
                        token.type = Token::IDENTIFIER;
                        token.str.ptr = &str[begin];
                        token.str.len = idx - begin;
                    }

                    --idx;
                }
                else
                {
                    error = true;
                    printf("%d: error: unexpected character '%c'\n", line, str[idx]);
                    break;
                }

                tokens.pushBack(token);
                break;
            }
        }

        ++idx;
    }

    tokens.pushBack({Token::LOX_EOF, line});

    return !error;
}

int main()
{
    Array<Token> tokens;

    const bool scanError = !scan(tokens, "-4 * ( 5 + 1) - 3 * 2 + 1");

    Parser parser(tokens);

    Expr expr;

    const bool parseError = !parser.parse(expr);

    if(!scanError && !parseError)
        interpret(expr);

    return 0;
}
