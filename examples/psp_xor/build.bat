psp-g++ -Os -g0 -Wall -std=c++17 -I. -IC:/pspsdk/psp/sdk/include -I../../src -L. -LC:/pspsdk/psp/sdk/lib -D_PSP_FW_VERSION=150 -c ../../src/*.cpp *.cpp -lstdc++ -lpspdebug -lpspdisplay -lpspge -lpspctrl -lpspsdk -lc -lpspnet -lpspnet_inet -lpspnet_apctl -lpspnet_resolver -lpsputility -lpspuser -lpspkernel -lpsppower -lpspgu -lpspgum -lc -lm
psp-gcc -Os -g0 -Wall -I. -IC:/pspsdk/psp/sdk/include -I../../src -L. -LC:/pspsdk/psp/sdk/lib -D_PSP_FW_VERSION=150 -o raw_eboot.elf *.o -lstdc++ -lpspdebug -lpspdisplay -lpspge -lpspctrl -lpspsdk -lc -lpspnet -lpspnet_inet -lpspnet_apctl -lpspnet_resolver -lpsputility -lpspuser -lpspkernel -lpsppower -lpspgu -lpspgum -lc -lm
psp-fixup-imports raw_eboot.elf
mksfo "PSP XOR Example" PARAM.SFO
psp-strip raw_eboot.elf -o strip_eboot.elf
pack-pbp EBOOT.PBP PARAM.SFO NULL NULL NULL NULL NULL strip_eboot.elf NULL
rm diwa.o psp_xor.o raw_eboot.elf strip_eboot.elf PARAM.SFO
mv EBOOT.PBP ../../dist/psp_xor.pbp