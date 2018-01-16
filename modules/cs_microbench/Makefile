TARGET=cs_microbench
LIBFAKEOSVDIR=../../libfakeosv

.PHONY: module
module: $(TARGET).so $(TARGET) usr.manifest

CXXFLAGS = -g -fPIC -I../../include

LDFLAGS += -lpthread -lboost_program_options

OBJS += main.o
OBJS += hopper.o
OBJS += cachestress.o
OBJS += ithrash.o
OBJS += icachelatency.o
OBJS += stagel1i.o
# cachestress code only works with O0
cachestress.o: CXXFLAGS += -O0
icachelatency.o: CXXFLAGS += -O0


HEADERS += ../../include/osv/stagesched.h
HEADERS += cs_microbench.hh
HEADERS += histogram.hh
HEADERS += work.hh
HEADERS += build.h

$(shell ./genbuild.sh)

$(TARGET).so: $(OBJS)
	$(CXX) -shared -o $@ $(LDFLAGS) $(OBJS)

$(TARGET): $(OBJS) $(TARGET).so $(LIBFAKEOSVDIR)/libfakeosv.so
	$(CXX) -o $@ $(LDFLAGS) -L./$(LIBFAKEOSVDIR) -lfakeosv $(OBJS)

.PHONY: usr.manifest
usr.manifest:
	echo '/$(TARGET).so: $${MODULE_DIR}/$(TARGET).so' > $@

$(OBJS): %.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $<

.PHONY: run
run_linux: $(TARGET)
	LD_LIBRARY_PATH=$(LIBFAKEOSVDIR) ./$(TARGET)

.PHONY: clean
clean:
	rm -f build.h
	rm -f *.o
	rm -f $(TARGET)
	rm -f $(TARGET).so
