- `nvcc -arch=$ARCH -keep -Xptxas -v -cubin file.cu -o file.cubin`
    - Tells nvcc to `-keep` results of the compilation in a `-cubin` file
    - `-Xptxas` assembles PTX to SASS
    - `-v` outputs useful information like shared memory, register usage
    - `-arch` is the specified architecture to compile for
- Simple makefile that does this and can clean up the directory afterward
```bash
ARCH=sm_86

test:
	nvcc -arch=$(ARCH) -Xptxas -v -keep -cubin file.cu -o file.cubin

clean:
	rm *.ii *.cubin *.c *.gpu *.module_id *.gpu *.ptx
```
