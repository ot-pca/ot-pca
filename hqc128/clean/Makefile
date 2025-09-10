# This Makefile can be used with GNU Make or BSD Make

LIB=lib/libhqc-128_clean.so
HEADERS=api.h code.h domains.h fft.h fips202.h gf.h gf2x.h hqc.h parameters.h parsing.h reed_muller.h reed_solomon.h randombytes.h shake_ds.h shake_prng.h vector.h 
OBJECTS=code.o fft.o fips202.o gf.o gf2x.o hqc.o kem.o parsing.o randombytes.o reed_muller.o reed_solomon.o shake_ds.o shake_prng.o vector.o 

CFLAGS=-O3 -Wall -Wextra -Wpedantic -Wshadow -Wvla -Werror -Wredundant-decls -Wmissing-prototypes -std=c99 -fPIC -I/common $(EXTRAFLAGS)
LDFLAGS=-shared

all: $(LIB)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(LIB): $(OBJECTS)
	mkdir -p lib
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(LIB)
