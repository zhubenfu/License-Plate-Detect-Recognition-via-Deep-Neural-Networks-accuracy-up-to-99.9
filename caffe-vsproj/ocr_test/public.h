#pragma once


//-------------------------------------------------------------------------------------------------
#ifdef  WIN32
	#ifndef IMPORT
	#define IMPORT __declspec(dllimport)
	#endif
	#ifndef EXPORT
	#define EXPORT __declspec(dllexport)
	#endif
	#ifndef API
	#define API __stdcall
	#endif
#else
	#ifndef IMPORT
	#define IMPORT
	#endif
	#ifndef EXPORT
	#define EXPORT
	#endif
	#ifndef API
	#define API
	#endif
#endif

//-------------------------------------------------------------------------------------------------
#ifndef interface
#define CINTERFACE
#define interface struct
#endif

//-------------------------------------------------------------------------------------------------
#ifndef min2
#define min2(a,b)  (((a)<(b)) ? (a) : (b)) 
#endif

//-------------------------------------------------------------------------------------------------
#ifndef max2
#define max2(a,b)  (((a)>(b)) ? (a) : (b))
#endif

//-------------------------------------------------------------------------------------------------
#ifndef _rect_
#define _rect_
struct rect{ int x0, y0, x1, y1; };
#define rectw(r) (r.x1-r.x0+1)
#define recth(r) (r.y1-r.y0+1)
#endif

//-------------------------------------------------------------------------------------------------
#ifndef byte
typedef unsigned char byte;
#endif

//-------------------------------------------------------------------------------------------------
#ifndef word
typedef unsigned short word;
#endif
