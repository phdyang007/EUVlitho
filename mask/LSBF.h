void randmask(int *mask1d, int NDIV, int cd, double fac1d);
void maskgen(int *mask2d, int NDIVX, int NDIVY)
{
  int* mask1dX=new int[NDIVX];
  int* mask1dY=new int[NDIVY];
  int cd = 60;
  double fac=1.;
  double fac1d=3.;
 int sum, a,space;
 space=rand()%2;
  sum=0; 
 while (sum < NDIVY-cd)
 { 
   a = cd*exp(fac*(rand()%1000)/1000.);
    if(space==1)
    {
       for(int i=sum;i<min(sum+a,NDIVY);i++)  
        for(int j=0;j<NDIVX;j++)
         mask2d[NDIVY*j+i]=0;
        space=0;
      }
      else
    {  
       randmask(mask1dX,NDIVX,cd,fac1d);
       for(int i=sum;i<min(sum+a,NDIVY);i++)  
       for(int j=0;j<NDIVX;j++)
        mask2d[NDIVY*j+i]=mask1dX[j];
        space=1;
      }
    sum=sum+a;
  }
 for(int i=sum;i<NDIVY;i++)
  {
   if(space == 0)
     for(int j=0;j<NDIVX;j++)
     mask2d[NDIVY*j+i]=0;
   else
   for(int j=0;j<NDIVX;j++)
   mask2d[NDIVY*j+i]=mask1dX[j];
  }

 space=rand()%2;
  sum=0; 
 while (sum < NDIVX-cd)
 { 
   a = cd*exp(fac*(rand()%1000)/1000.);
    if(space==1)
    {
//       for(int i=sum;i<min(sum+a,NDIV);i++)  
//        for(int j=0;j<NDIV;j++)
//         mask2d[NDIV*j+i]=0;
        space=0;
      }
      else
    {  
       randmask(mask1dY,NDIVY,cd,fac1d);
       for(int i=sum;i<min(sum+a,NDIVX);i++)  
       for(int j=0;j<NDIVY;j++)
        mask2d[NDIVY*i+j]=max(mask1dY[j],mask2d[NDIVY*i+j]);
        space=1;
      }
    sum=sum+a;
  }
 for(int i=sum;i<NDIVX;i++)
  {
   if(space == 0)
//     for(int j=0;j<NDIV;j++)
//     mask2d[NDIV*j+i]=0;
         space=1;
   else
   for(int j=0;j<NDIVY;j++)
    mask2d[NDIVY*i+j]=max(mask1dY[j],mask2d[NDIVY*i+j]);
  }
}

void randmask(int *mask1d, int NDIV, int cd, double fac1d)
{
 int sum, a,line;
 line=rand()%2;
 sum=0;    
 while (sum < NDIV-cd)
 { 
 a=cd*exp(fac1d*(rand()%1000)/1000.);
 for(int i=sum;i<min(sum+a,NDIV);i++)
 {
     if(line == 1)
      mask1d[i] = 1;
      else
      mask1d[i] = 0;
     }
    if(line == 1)
     line = 0;
    else
     line = 1;
    sum = sum + a;
  }
 for(int i=sum;i<NDIV;i++)
  {
   if(line == 0)
    mask1d[i] = 1;
   else
    mask1d[i] = 0;
  }
}

