CXX := clang++
CXXFLAGS := -fclangir -Wno-unused-value -emit-cir

SRC := vecmul.cpp
CIR := $(SRC:.cpp=.cir.mlir)

.PHONY: all emit-cir clean

all: emit-cir

emit-cir: $(CIR)

$(CIR): $(SRC)
		$(CXX) $(CXXFLAGS) -o $@ $<

clean:
		rm -f *.cir.mlir
