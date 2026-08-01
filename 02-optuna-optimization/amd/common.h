#ifndef GLX
#define GLX 512
#endif 

#ifndef GLY
#define GLY 512
#endif 

#ifndef MAXITER
#define MAXITER 10000
#endif

#ifndef DUMP
#define DUMP 0
#endif

#ifndef DUMPSTEP
#define DUMPSTEP 200
#endif


#define TB 72.0
#define TH 212.0
#define TC 32.0

#define HX 1
#define HY 1

#define GX (HX+GLX+HX)
#define GY (HY+GLY+HY)

#define K 16

int GLYH2 = (int) (GLY/2); // the middle of the grid

void init(double *grid){

    for(int i=0; i<GY; i++){
        for(int j=0; j<GX; j++)
        {
            int index = i*GX + j; // corrisponde all'indice [i,j]
            grid[index] = TC; // set the grid including ghost borders to 32°
        }
    }

    // init vertical gradient
    // begin the computation of the gradient from 1 to GLY/2
    for( int i = HY; i < HY+GLYH2; i++){
        for(int j = (GX-HX-1-K); j < (GX-HX); j++){
            grid[i*GX+j] = TC + ((double)(TH*2*(i-HY))/((double)(GLY)));
        } // temperatura cresce (dall'alto verso il centro)
    }
    // continue from GLY/2 to the end of the grid
    for (int i = HY+GLYH2; i < HY + GLY; i++){
        for(int j = (GX-HX-1-K); j<(GX-HX); j++){
            grid[i*GX+j] = TC + ((double)(TH*2*(GY-(i-HY))) / ((double)(GLY)) );
        } // temperatura decresce (dal centro al basso)
    }
}

// GLX x GLY is the size of the physical grid
// HX and HY are the sizes of the ghost borders
// GX x GY is the size of the grid including ghost borders

void dump(double *grid,const char *filename){
    FILE *file = fopen(filename,"w");

    if(file == NULL) {
        fprintf(stderr,"Error opening output file\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 1; i<GY-1; i++){
        fwrite((grid+(i*GX)+1), sizeof(double), GLX, file);
    }
    fclose(file);
}

// ################################

