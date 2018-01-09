.PHONY: module

VERSION :=$(shell cd src && git describe --tags --dirty)
BASEDIR := "${CURDIR}"
BUILDDIR="$(BASEDIR)/build"
ROOTDIR="$(BASEDIR)/install"
SRCDIR="$(BASEDIR)/src"
LIBFAKESTDDIR="$(BASEDIR)/../../modules/cs_microbench/libfakestd"
CFLAGS="-fPIC"
CXXFLAGS="-fPIC -fpermissive"

module: usr.manifest

usr.manifest: install
	printf "\
/etc/**: $(ROOTDIR)/etc/**\n\
/usr/share/**: $(ROOTDIR)/usr/share/**\n\
/usr/lib/**: $(ROOTDIR)/usr/lib/**\n\
/usr/data/**: $(ROOTDIR)/usr/data/**\n\
/usr/bin/mysqld: $(BUILDDIR)/sql/mysqld\n\
" > usr.manifest

.PHONY: install reinstall
install: install.depend
reinstall:
	rm install.depend
	make install

install.depend: build.depend
	rm -rf $(ROOTDIR)
	mkdir -p $(ROOTDIR)/etc # includes install dir
	$(MAKE) -C $(BUILDDIR) install DESTDIR=$(ROOTDIR)
	cp $(BASEDIR)/my.cnf $(ROOTDIR)/etc/
	LD_LIBRARY_PATH=$(LIBFAKESTDDIR) \
        ./configure_install.bash $(BASEDIR) $(ROOTDIR) $(PRESEED)
	touch install.depend


.PHONY: build
build: build.depend

build.depend:
	mkdir -p $(BUILDDIR)
	cd $(BUILDDIR); \
	cmake \
		-DCMAKE_INSTALL_PREFIX:PATH=/usr \
		-DCMAKE_CXX_FLAGS='-fpermissive -fPIC' \
		-DCMAKE_C_FLAGS='-fPIC' \
        -DCMAKE_CXX_LINK_FLAGS='-L$(LIBFAKESTDDIR)' \
		-DWITH_UNIT_TESTS=OFF \
		-DWITH_NDBCLUSTER=OFF \
		-DWITH_NDBCLUSTER_STORAGE_ENGINE=OFF \
		-DWITH_PROTOBUF=OFF \
		-DWITH_EMBEDDED_SERVER=OFF \
		-DWITH_EMBEDDED_SHARED_LIBRARY=OFF \
		-DWITH_MYSQLD_LDFLAGS="-pie" \
		$(SRCDIR)
	LD_LIBRARY_PATH=$(LIBFAKESTDDIR) \
		$(MAKE) -C $(BUILDDIR)
	touch build.depend
	
.PHONY: clean
clean:
	rm -rf build install usr.manifest *.depend

.PHONY: cleanall
cleanall: clean

