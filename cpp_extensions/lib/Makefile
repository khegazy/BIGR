BASECODE=../src
BASEHPP=../include
CXX=g++ -shared
CXXFLAGS=-std=c++11 -Wall -fPIC -Wno-unused-local-typedefs -I$(BASEHPP)
LINKLIBS=

CPP_FILES := $(wildcard $(BASECODE)/*.cpp)
HPP_FILES := $(wildcard $(BASEHPP)/*.h)
OBJ_FILES := $(notdir $(CPP_FILES:%.cpp=%.o))
SO_FILES := $(OBJ_FILES:%.o=%.so)
T_FILES := $(patsusbst 'src', 'lib', $(wildcard $(BASECODE)/*.cpp))
STD := -std=c++0x

$(info ${OBJ_FILES})
$(info ${CPP_FILES})
$(info ${T_FILES})

vpath %.cpp $(sort $(dir $(CPP_FILES)))

all: $(SO_FILES)

.PRECIOUS: %.o 

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.so: %.o
	$(CXX) -o $@ $^ $(LINKLIBS)

clean:
	rm -f *.o *.so
