#include<stdio.h>
#include<stdlib.h>
#define ll long long int

ll upperBound(int *A,ll low, ll high,ll K){
    ll mid = ( low + high ) / 2;
    while(low <= high){
        mid = ( low + high ) / 2 ; 
        if(A[mid] > K && ( mid == 0 || A[mid-1] <= K ))
            return mid ;
        else if(A[mid] > K) 
            high = mid - 1 ;
        else
            low = mid + 1 ;
    }
    return mid ;
}


ll lowerBound(int *A,ll low, ll high,ll K){
    ll mid = ( low + high ) / 2;
    while(low <= high){
        mid = ( low + high ) / 2 ; 
        if(A[mid] >= K && ( mid == 0 || A[mid-1] < K ))
            return mid ;
        else if(A[mid] >= K) 
            high = mid - 1 ;
        else
            low = mid + 1 ;
    }
    return mid ;
}