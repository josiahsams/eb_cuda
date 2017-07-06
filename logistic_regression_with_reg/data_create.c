
#include <stdio.h>

int main(){

FILE *fp1;
fp1 = fopen("input", "w");
for (int i=0; i<500000; i++){
        //fprintf(fp1, "%lf  \t %lf \t %d \n", (i*0.1), (i+0.1), 1);
	for (int j=0; j<400; j++){
		fprintf(fp1, "%lf \t",  (i+j+0.1));
	}
	fprintf(fp1, "%d \n", 1);
}

for (int i=0; i<500000; i++){
        //fprintf(fp1, "%lf  \t %lf \t %d \n", (-i*0.1), -(i+0.1), 0);
	for (int j=0; j<400; j++){
        fprintf(fp1, "%lf \t",  -(i+j+0.1));
	}
	fprintf(fp1, "%d \n", 0);
}

}
