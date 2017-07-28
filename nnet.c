#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/*for (int j=0; j<n_hidden; j++){
	sum_hidden[j] = weight_hidden[0][j];
	for (int i=0; i<n_input; i++){
		sum_hidden[j] += input[i] * weight_hidden[i][j];
	}
	hidden[j] = 1.0/(1.0 + exp(-sum_hidden[j])
}


for (int k=0; k<n_output; k++){
	sum_output[k] = weight_hidden_output[0][k];
	for (int j=0; j<n_hidden; j++){
		sum_output[k] += hidden[j] * weight_hidden_output[j][k];
	}
	output[k] = 1.0/(1.0 + exp(-sum_output[k]));
}

float error = 0.0;
for (int p=0; p<n; p++){
	for (int k=0; k<n_output; k++){
		error += 0.5 * (target[p][k] - output[p][k]) * (target[p][k] - output[p][k]);
	}
}*/


#define NumInput 2
#define NumHidden 4
#define NumOutput 1
#define NumPattern 4
#define eta 0.5
#define alpha 0

#include <time.h>

int main(){

double Input[NumPattern][NumInput] ;
double Hidden[NumPattern][NumHidden] ;
double WeightIH[NumInput][NumHidden] ;
double DeltaWeightIH[NumInput][NumHidden] ;
double WeightHO[NumHidden][NumOutput] ;
double DeltaWeightHO[NumInput][NumHidden] ;
double SumH[NumPattern][NumHidden];
double SumO[NumPattern][NumOutput];
double Output[NumPattern][NumOutput];
double Target[NumPattern][NumOutput];
double DeltaO[NumOutput];
double DeltaH[NumHidden];
double SumDOW[NumHidden];


Input[0][0] = 0;
Input[0][1] = 0;
Input[1][0] = 0;
Input[1][1] = 1;
Input[2][0] = 1;
Input[2][1] = 0;
Input[3][0] = 1;
Input[3][1] = 1;

Target[0][0] = 0;
Target[1][0] = 1;
Target[2][0] = 1;
Target[3][0] = 0;



/*for (int i=0; i<2500; i++){
	for (int j=0; j<20; j++){
		Input[i][j] = i+j;
	}
}


for (int i=0; i<2500; i++){
	if ((i%2)==0) Target[i][0] = 0;
	else Target[i][0] = 1;
}



for (int i=0; i<1250; i++){
        //fprintf(fp1, "%lf  \t %lf \t %d \n", (i*0.1), (i+0.1), 1);
        for (int j=0; j<20; j++){
                Input[i][j] =  (i+j+0.1);
        }
        Target[i][0] = 1;
}

for (int i=1250; i<2500; i++){
        //fprintf(fp1, "%lf  \t %lf \t %d \n", (-i*0.1), -(i+0.1), 0)
        for (int j=0; j<20; j++){
        	Input[i][j] = -(i+j+0.1);
        }
        Target[i][0] = 0;
}
*/

clock_t start, end;
double time_used;
start = clock();

for(int j = 0 ; j < NumHidden ; j++ ) {         /* initialize WeightIH and DeltaWeightIH */

    for(int i = 0 ; i < NumInput ; i++ ) {
        DeltaWeightIH[i][j] = 0.0 ;
        WeightIH[i][j] = 2.0 * ( rand()%NumPattern - 0.5 ) * 0.02 ;
	//WeightIH[i][j] = 0.0;
	//printf("%lf \t", WeightIH[i][j]);
    }
	//printf("\n");

}

for(int k = 0 ; k < NumOutput ; k ++ ) {         /* initialize WeightHO and DeltaWeightHO */

    for(int j = 0 ; j < NumHidden ; j++ ) {
        DeltaWeightHO[j][k] = 0.0 ;
        WeightHO[j][k] = 2.0 * ( rand()%NumPattern - 0.5 ) * 0.01 ;
    }

}


for (int i=0; i<10; i++){
	printf("%lf \n", WeightHO[i][0]);
}


//printf("%lf \n", WeightIH[1][0]);

for (int iter=0; iter<2; iter++){
	double Error = 0.0 ;
	for(int p = 0 ; p < NumPattern ; p++ ) {         /* repeat for all the training patterns */
		double Error = 0.0;
		for(int j = 0 ; j < NumHidden ; j++ ) {         /* compute hidden unit activations */
        		SumH[p][j] = WeightIH[0][j] ;
        		for(int i = 0 ; i < NumInput ; i++ ) {
            			SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
	    			//printf("%lf \t", SumH[p][j]);
        		}
        		Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
			//printf("\n");
			//printf("%lf \n", Hidden[p][j]);
	}

	/*for (int i=0; i<10; i++){
		printf("%lf \n", Hidden[0][i]);
	}*/

	//printf("ERRORRR : %lf \n", Error);

	for(int k = 0 ; k < NumOutput ; k++ ) {         /* compute output unit activations and errors */
        	SumO[p][k] = WeightHO[0][k] ;
        	for(int j = 0 ; j < NumHidden ; j++ ) {
            		SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
        	}
        	Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;
		//printf("%lf \n", Output[p][k]);
		//printf("%lf \n", Target[p][k]);
        	Error += (Target[p][k] - Output[p][k]); //* (Target[p][k] - Output[p][k]) ;
		//printf("%lf \n", Error);
        	DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;
	}

	

	/*for (int i=0; i<10; i++){
      printf("%lf \n", Output[0][i]);
	}*/
	//printf("ERRORRR : %lf \n", Error);

	for(int j = 0 ; j < NumHidden ; j++ ) {         /* 'back-propagate' errors to hidden layer */
        	SumDOW[j] = 0.0 ;
        	for(int k = 0 ; k < NumOutput ; k++ ) {
            		SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
        	}
        	DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
    	}

    	for(int j = 0 ; j < NumHidden ; j++ ) {         /* update weights WeightIH */
        	DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
        	WeightIH[0][j] += DeltaWeightIH[0][j] ;
        	for(int i = 0 ; i < NumInput ; i++ ) {
            		DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
            		WeightIH[i][j] -= DeltaWeightIH[i][j] ;
			//printf("%lf \t", WeightIH[i][j]);
        	}
		//printf("\n");
    	}

	for(int k = 0 ; k < NumOutput ; k ++ ) {         /* update weights WeightHO */
        	DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
        	WeightHO[0][k] += DeltaWeightHO[0][k] ;
        	for(int j = 0 ; j < NumHidden ; j++ ) {
            		DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
            		WeightHO[j][k] -= DeltaWeightHO[j][k] ;
        	}
	}

	}
	//printf("ERROR : %lf", Error);
}

//printf("%lf \n", WeightIH[1][0]);





end = clock();
time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
printf("Time taken for copy in : %f \n", time_used);


for (int i=0; i<10; i++){
        printf("%lf \n", WeightIH[0][i]);
}


 for (int i=0; i<10; i++){
        printf("%lf \n", WeightHO[0][i]);
}
}
