CXX = g++
CPPFLAGS = -std=c++17 -O3 -march=native -fopenmp
RM = rm -f
OBJS = util.o Graph.o Interval.o 
all: $(OBJS)
	$(CXX) $(CPPFLAGS) test.cpp Graph.o Interval.o util.o -o test

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) test
