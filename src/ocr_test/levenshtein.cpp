#include <stdlib.h>
#include <string.h>
#include "levenshtein.h"

static int minimum(int a,int b,int c)
/*Gets the minimum of three values*/
{
  int min=a;
  if(b<min)
    min=b;
  if(c<min)
    min=c;
  return min;
}

int levenshtein_distance(char *s, int n, char*t, int m, int noop)
/*Compute levenshtein distance between s and t*/
{
  //Step 1
  int k,i,j,cost,*d,distance;
  if(n!=0&&m!=0)
  {
    d=(int*)malloc((sizeof(int))*(m+1)*(n+1));
    m++;
    n++;
    //Step 2	
    for(k=0;k<n;k++)
	d[k]=k;
    for(k=0;k<m;k++)
      d[k*n]=k;
    //Step 3 and 4	
    for(i=1;i<n;i++)
      for(j=1;j<m;j++)
	{
        //Step 5
        if(s[i-1]==t[j-1])
          cost=0;
        else
          cost=1;
        //Step 6			 
        d[j*n+i]=minimum(d[(j-1)*n+i]+1,d[j*n+i-1]+1,d[(j-1)*n+i-1]+cost);
      }
    distance=d[n*m-1];
    free(d);
    return distance;
  }
  else 
    return -1; //a negative return value means that one or both strings are empty.
}

int levenshtein(char * s1, int l1, char * s2, int l2, int threshold) {
  int * prev_row, * curr_row;
  int col, row;
  int curr_row_min, result;
  int offset = 0;

  /* Do the expensive calculation on a subset of the sequences, if possible, by removing the common prefix. */

  while (s1[offset] == s2[offset]) {
    offset++;
  }

  /* Do the expensive calculation on a subset of the sequences, if possible, by removing the common postfix. */

  while ((l1-1 > offset) && (l2-1 > offset) && (s1[l1-1] == s2[l2-1])) {
    l1--;
    l2--;
  }

  l1 -= offset;
  l2 -= offset;

  /* The Levenshtein algorithm itself. */

  /*       s1=              */
  /*       ERIK             */
  /*                        */
  /*      01234             */
  /* s2=V 11234             */
  /*    E 21234             */
  /*    E 32234             */
  /*    N 43334 <- prev_row */
  /*    S 54444 <- curr_row */
  /*    T 65555             */
  /*    R 76566             */
  /*    A 87667             */

  /* Allocate memory for both rows */

  prev_row	= (int*)malloc(l1+1);
  curr_row	= (int*)malloc(l1+1);

  if ((prev_row == NULL) || (curr_row == NULL)) {
      return -1;
  }

  /* Initialize the current row. */

  for (col=0; col<=l1; col++) {
    curr_row[col] = col;
  }

  for (row=1; row<=l2; row++) {
    /* Copy the current row to the previous row. */

    memcpy(prev_row, curr_row, sizeof(int)*(l1+1));

    /* Calculate the values of the current row. */

    curr_row[0]		= row;
    curr_row_min	= row;

    for (col=1; col<=l1; col++) {
      /* Equal (cost=0) or substitution (cost=1). */

      curr_row[col]	= prev_row[col-1] + ((s1[offset+col-1] == s2[offset+row-1]) ? 0 : 1);

      /* Insertion if it's cheaper than substitution. */

      if (prev_row[col]+1 < curr_row[col]) {
        curr_row[col] = prev_row[col]+1;
      }

      /* Deletion if it's cheaper than substitution. */

      if (curr_row[col-1]+1 < curr_row[col]) {
        curr_row[col] = curr_row[col-1]+1;
      }

      /* Keep track of the minimum value on this row. */

      if (curr_row[col] < curr_row_min) {
        curr_row_min	= curr_row[col];
      }
    }

    /* Return nil as soon as we exceed the threshold. */

    if (threshold > -1 && curr_row_min >= threshold) {
      free(prev_row);
      free(curr_row);

      return -1;
    }
  }

  /* The result is the last value on the last row. */

  result = curr_row[l1];

  free(prev_row);
  free(curr_row);

  return result;
}