CXX      := g++
SRCDIR   := src
INCLUDES := -I$(SRCDIR)
WARNINGS := -Wall -Wextra
CXXFLAGS := -std=c++20 $(WARNINGS) -g -O2 -MMD -MP $(INCLUDES)
LDFLAGS  := -lsfml-graphics -lsfml-window -lsfml-system

TARGET  := nn
BUILD   := build

# Recursive, so a new src/<group>/ needs no Makefile change.
SRCS := $(shell find $(SRCDIR) -name '*.cpp')
OBJS := $(SRCS:%.cpp=$(BUILD)/%.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all run clean release

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Objects mirror the source tree under build/, so same-named files in
# different groups cannot collide.
$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

release: CXXFLAGS := -std=c++20 $(WARNINGS) -O2 -DNDEBUG -MMD -MP $(INCLUDES)
release: clean $(TARGET)

clean:
	rm -rf $(BUILD) $(TARGET)

-include $(DEPS)
