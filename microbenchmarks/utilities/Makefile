# Used to facilitate a "godbolt"-like experience in neovim by invoking `make`
# with a selection and opening the resulting .ptx files in a new buffer with a
# neovim command
# - See godbolt: https://godbolt.org/

ARCH=sm_86

test:
	nvcc -arch=$(ARCH) -Xptxas -v -keep -cubin file.cu -o file.cubin

clean:
	rm *.ii *.cubin *.c *.gpu *.module_id *.gpu *.ptx
