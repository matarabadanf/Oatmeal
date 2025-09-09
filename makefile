# Compiler and flags
CC = gcc # compiler 
CFLAGS = -O2 -Wall -Iinclude # compiler flags at object phase
LDFLAGS = -llapacke -llapack -lblas -lm # link flags at linking step 

################### Default ######################
# Source and object files
SRC = $(shell find src -name '*.c') # all source files with a .c extension
OBJ = $(SRC:.c=.o) # all object files created from all source files changing .c by .o 
MAIN = src/program.exe # main executable final name 

EXES = $(MAIN)

all: $(MAIN) # all depends on main. 

$(MAIN): src/program.o $(filter-out src/program.o,$(OBJ)) # main depends on program.o and all the object files that are not program.o. Next line compiles all dependencies ($^) with the name $@ (-o) using the linking flags 
	$(CC) $^ -o $@ $(LDFLAGS) 


##################### tests ###################### 

# Test files (exclude run_tests.c)
TEST_SRC = $(filter-out tests/run_tests.c,$(shell find tests -name '*.c'))
TEST_OBJ = $(TEST_SRC:.c=.o)
TEST_EXES = $(TEST_OBJ:.o=.exe)

# Add run_tests explicitly
ALL_TEST_EXES = $(TEST_EXES) tests/run_tests.exe

tests: $(ALL_TEST_EXES)

# Compile all .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Special rule for run_tests (no OBJ linked)
tests/run_tests.exe: tests/run_tests.o
	$(CC) $< -o $@ $(LDFLAGS)

# All other tests link with OBJ
tests/%.exe: tests/%.o $(OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)


#################### Clean #######################
clean:
	rm -f $(EXES) $(ALL_TEST_EXES) tests/run_tests.o src/program.exe

# Clean everything including object files
clean-all:
	rm -f $(OBJ) $(EXES) $(TEST_OBJ) $(ALL_TEST_EXES) tests/run_tests.o src/program.exe

# Show what will be built
debug:
	@echo "Source files: $(SRC)"
	@echo "Object files: $(OBJ)"
	@echo "Executables: $(EXES)"
	@echo "Test source files: $(TEST_SRC)"
	@echo "Test object files: $(TEST_OBJ)"
	@echo "Test executables: $(ALL_TEST_EXES)"

.PHONY: all tests clean clean-all debug
