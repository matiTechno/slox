all:
	g++ -g -Wall -Wextra -pedantic -std=c++14 -o slox main.cpp \
            -Wno-missing-field-initializers -Wno-switch -fno-exceptions -fno-rtti
