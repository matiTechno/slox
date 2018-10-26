COMMON=g++ -Wall -Wextra -pedantic -std=c++14 -o slox main.cpp \
       -Wno-missing-field-initializers -Wno-switch -fno-exceptions -fno-rtti

debug:
	$(COMMON) -g

rel:
	$(COMMON) -O2
