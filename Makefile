default:
	nvcc main.cu kernel.cu

clean:
	rm a.out