
CXX := g++
CXXFLAGS := -fPIC -I../include

OBJS += stagesched.o

HEADERS += ../include/osv/stagesched.h

libfakeosv.so: $(OBJS)
	$(CXX) -shared -lpthread -o $@ $(OBJS)

libfakeosv.a: $(OBJS)
	ar rcs $@ $(OBJS)

%.o: %.cc $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -rf *.o *.a *.so
