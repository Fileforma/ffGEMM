#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include "cJSON.h" 
#define MAX_FILENAME_LENGTH 256
enum SAFETENSORS_DTYPES
{
	SAFETENSORS_F64 = 0,
	SAFETENSORS_F32,
	SAFETENSORS_F16,
	SAFETENSORS_BF16,
	SAFETENSORS_I64,
	SAFETENSORS_I32,
	SAFETENSORS_I16,
	SAFETENSORS_I8,
	SAFETENSORS_U8,
	SAFETENSORS_BOOL,
	SAFETENSORS_NUM_DTYPES
};
char dataTypes_String_Safetensors[SAFETENSORS_NUM_DTYPES][20] ={"F64","F32","F16","BF16","I64","I32","I16","I8","U8","BOOL"};

int GetSafetensorSize(int dtype)
{
	switch(dtype) {
	case SAFETENSORS_F64:  return 8;
	case SAFETENSORS_F32:  return 4;
	case SAFETENSORS_F16:  return 2;
	case SAFETENSORS_BF16: return 2;
	case SAFETENSORS_I64:  return 8;
	case SAFETENSORS_I32:  return 4;
	case SAFETENSORS_I16:  return 2;
	case SAFETENSORS_I8:   return 1;
	case SAFETENSORS_U8:   return 1;
	case SAFETENSORS_BOOL: return 1; // TODO check if this is right
	}
	return 0;
}
void SortFileNames(char **fileNames, int fileCount)
{
	char temporary[MAX_FILENAME_LENGTH] = {0};
	for(int k = 0; k < fileCount; k++)
	{
		for(int j = k + 1; j < fileCount; j++)
		{
			if(strcasecmp(fileNames[k], fileNames[j]) > 0)
			{
				strcpy(temporary, fileNames[k]);
				strcpy(fileNames[k], fileNames[j]);
				strcpy(fileNames[j], temporary);
				memset(temporary,0,MAX_FILENAME_LENGTH);
			}
		}
	}
}
char *CreateStorageDirectory(char *path)
{
	char outputFolder[MAX_FILENAME_LENGTH] = {0};
	snprintf(outputFolder, sizeof(outputFolder), "%s",path);
	DIR *directory = opendir(outputFolder);
	if(directory == NULL)
	{
		int outputFolderCreationSuccess = mkdir(outputFolder,0777);assert(outputFolderCreationSuccess == 0);
	}
	else
	{
		closedir(directory);
	}
	char *result = calloc(MAX_FILENAME_LENGTH, sizeof(char));
	for(int i = 0; i < MAX_FILENAME_LENGTH; i++)
	{
		result[i] = outputFolder[i];
	}
	return result;
}
int CountNumberOfFilesInDirectory(char *path)
{
	int result = 0;
	DIR *directory = opendir(path);assert(directory != NULL);
	struct dirent *directoryEntry;
	while((directoryEntry = readdir(directory)) != NULL)
	{
		if(directoryEntry->d_name[0] != '.')
		{
			int fileNameLength = (int)sizeof(directoryEntry->d_name);assert(fileNameLength <= MAX_FILENAME_LENGTH);result++;
		}
	}
	closedir(directory);
	return result;
}
char **GetFileNamesInDirectory(char *path, int *fileCount)
{	
	/*Count files and ensure length <= 256*/
	*fileCount = CountNumberOfFilesInDirectory(path);
	/*Array to store file names*/
	char **result = malloc(*fileCount * sizeof(char*));
	for(int i = 0; i < *fileCount; i++)
	{
		result[i] = calloc(MAX_FILENAME_LENGTH, sizeof(char));
	}
	/*Get file names*/
	DIR *directory = opendir(path);assert(directory != NULL);
	struct dirent *directoryEntry;
	int currentIndex = 0;
	while((directoryEntry = readdir(directory)) != NULL)
	{
		if(directoryEntry->d_name[0] != '.')
		{
			assert(strlen(directoryEntry->d_name) < MAX_FILENAME_LENGTH);
			for(int i = 0; i < strlen(directoryEntry->d_name); i++)
			{
				result[currentIndex][i] = directoryEntry->d_name[i];
			}
			currentIndex++;
		}
	}
	SortFileNames(result, *fileCount);
	closedir(directory);
	return result;
}

void PrintFileNames(int fileCount, char **fileNames)
{
	for(int i = 0; i < fileCount; i++)
	{
		printf("%3d %s\n", i, fileNames[i]);
	}
}

void DestroyFileNames(int fileCount, char **fileNames)
{
	for(int i = 0; i < fileCount; i++)
	{
		free(fileNames[i]);
	}
	free(fileNames);
}

void PrintParameters(int tensorParameterSize, int *tensorParameter_Dtype, size_t *tensorParameter_Offsets, int *tensorParameter_MatrixHeight, int *tensorParameter_MatrixWidth)
{
	for(int i = 0; i < tensorParameterSize; i++)
	{
		printf("%3d : %d %ld [%5d %5d]\n", i, tensorParameter_Dtype[i], tensorParameter_Offsets[i], tensorParameter_MatrixHeight[i], tensorParameter_MatrixWidth[i]);
	}
}

size_t GetFileSize(char *fileName){FILE *fp = fopen(fileName, "rb");assert(fp != NULL);fseek(fp, 0L, SEEK_END);size_t currentFileSize = ftell(fp);rewind(fp);fclose(fp);return currentFileSize;}
void FindTensorParameters(cJSON *tensorData, size_t headerLength, int tensorParameterSize, int *tensorParameter_Dtype, size_t *tensorParameter_Offsets, int *tensorParameter_MatrixHeight, int *tensorParameter_MatrixWidth)
{
	int currentIndex = 0;int currentMatrixIndex = 0;size_t currentOffset = 0;int matrixDimensions = 0;
	cJSON *item = NULL;cJSON *offset = NULL;cJSON *dtype = NULL;cJSON *data_offsets = NULL;cJSON *shape = NULL;cJSON *eachShape = NULL;
	cJSON_ArrayForEach(item, tensorData)
	{
		dtype = cJSON_GetObjectItem(item, "dtype");data_offsets = cJSON_GetObjectItem(item, "data_offsets");
		shape = cJSON_GetObjectItem(item, "shape");
		if(dtype && data_offsets && shape)
		{
			//printf("Key: %s\n", item->string);printf("  dtype: %s\n", dtype->valuestring);printf("  data_offsets: ");
			matrixDimensions = cJSON_GetArraySize(shape);
			assert(matrixDimensions > 0);assert(matrixDimensions < 3);assert(currentIndex < tensorParameterSize);
			if(matrixDimensions == 1){tensorParameter_MatrixHeight[currentIndex] = 1;cJSON_ArrayForEach(eachShape, shape){tensorParameter_MatrixWidth[currentIndex] = eachShape->valueint;}}else{currentMatrixIndex = 0;cJSON_ArrayForEach(eachShape, shape){if(currentMatrixIndex == 0){tensorParameter_MatrixHeight[currentIndex] = eachShape->valueint;}else{tensorParameter_MatrixWidth[currentIndex] = eachShape->valueint;}currentMatrixIndex += 1;}}
			
			
			//printf("%3d : \n", matrixDimensions);
			for(int i = 0; i < SAFETENSORS_NUM_DTYPES; i++){if(strcmp(dtype->valuestring, dataTypes_String_Safetensors[i]) == 0){tensorParameter_Dtype[currentIndex] = i;break;}}
			cJSON_ArrayForEach(offset, data_offsets){currentOffset = (size_t) offset->valuedouble;break;}
			tensorParameter_Offsets[currentIndex] = headerLength + currentOffset + 8;
			currentIndex += 1;
		}
	}       
}



float bf16_to_float(uint8_t byte1, uint8_t byte2) {
    // Combine the two bytes into a 16-bit BF16 number
    uint16_t bf16 = ((uint16_t)byte1 << 8) | byte2;

    // Convert the BF16 number to a 32-bit float
    // - Sign bit: copy from bf16 (bit 15)
    // - Exponent: copy from bf16 (bits 14-7)
    // - Mantissa: copy bf16 (bits 6-0) into the top 7 bits of the 23-bit mantissa,
    //   and pad the remaining 16 bits with 0.
    
    // Prepare the 32-bit floating-point number by shifting the BF16 bits to the right position
    uint32_t sign = (bf16 & 0x8000) << 16;        // Sign bit (bit 31)
    uint32_t exponent = (bf16 & 0x7F80) << 16;    // Exponent (bits 30-23)
    uint32_t mantissa = (bf16 & 0x007F) << 16;    // Mantissa (bits 22-16)

    // Combine the sign, exponent, and mantissa to form a 32-bit IEEE 754 float
    uint32_t float_bits = sign | exponent | mantissa;

    // Reinterpret the bits as a float and return the result
    float result;
    *((uint32_t*)&result) = float_bits;
    return result;
}



void InvestigateTensors(size_t fileSize, unsigned char *fileData, int tensorParameterSize, int *tensorParameter_Dtype, size_t *tensorParameter_Offsets)
{
	assert(fileData != NULL);assert(tensorParameter_Dtype != NULL);assert(tensorParameter_Offsets != NULL);assert(tensorParameterSize > 1);assert(tensorParameter_Offsets[tensorParameterSize-1] < fileSize);
	int currentByteIndex = 0;
	int alphabet0[256] = {0};for(int j = 0; j < 256; j++){alphabet0[j] = j;}
	size_t frequency[256] = {0};unsigned char newSymbol=0;
	FILE *fr = fopen("Output/temp", "wb");assert(fr != NULL);
	for(int i = 0; i < tensorParameterSize-1; i++)
	{
		printf("%ld %ld : %ld %ld %d\n\n", tensorParameter_Offsets[i],tensorParameter_Offsets[i+1], tensorParameter_Offsets[i+1] - tensorParameter_Offsets[i], fileSize - tensorParameter_Offsets[i+1], GetSafetensorSize(tensorParameter_Dtype[i]));	
		currentByteIndex = 0;
		for(int j = tensorParameter_Offsets[i]; j < tensorParameter_Offsets[i+1]; j++)
		{
			//printf("%d : %3u ", currentByteIndex, fileData[j]);
			currentByteIndex += 1;
			if(currentByteIndex == GetSafetensorSize(tensorParameter_Dtype[i]))
			{
				//frequency[fileData[j]] += 1;
				currentByteIndex = 0;
				//printf("\n");
			}
			//if(currentByteIndex % GetSafetensorSize(tensorParameter_Dtype[i]) == 0)			
		}
	}
	//Get Final tensor
	fclose(fr);
}

void EncodeSingleFile(char *inputFileName, char *outputFileName)
{
	size_t fileSize = GetFileSize(inputFileName);
	FILE *fp = fopen(inputFileName, "rb");assert(fp != NULL);
	FILE *fr = fopen(outputFileName, "wb");assert(fr != NULL);
	int fileNumber = fileno(fp);
	unsigned char *fileData = mmap(NULL,fileSize, PROT_READ|PROT_WRITE, MAP_PRIVATE, fileNumber, 0);assert(fileData != NULL);
	
	/*Read HeaderLength(1st 8 bytes in reverse)*/
	size_t headerLength = 0;for(int i = 7; i >= 0; i--){headerLength <<= 8;headerLength += fileData[i];}assert(headerLength >= 0);
	assert(fileData[8] == '{');
	
	
	cJSON *tensorData = cJSON_ParseWithLength(fileData+8, headerLength);
	int tensorParameterSize = cJSON_GetArraySize(tensorData)-1;
	char *formatted_json = cJSON_Print(tensorData);if(formatted_json != NULL){printf("%s\n", formatted_json);free(formatted_json);  }
	assert(tensorParameterSize > 0);
	printf("%ld %d\n", headerLength, tensorParameterSize);
	
	int tensorParameter_Dtype[tensorParameterSize];
	size_t tensorParameter_Offsets[tensorParameterSize];
	int tensorParameter_MatrixHeight[tensorParameterSize];
	int tensorParameter_MatrixWidth[tensorParameterSize];
	FindTensorParameters(tensorData, headerLength, tensorParameterSize, tensorParameter_Dtype, tensorParameter_Offsets, tensorParameter_MatrixHeight, tensorParameter_MatrixWidth);
	PrintParameters(tensorParameterSize, tensorParameter_Dtype, tensorParameter_Offsets, tensorParameter_MatrixHeight, tensorParameter_MatrixWidth);

	//InvestigateTensors(fileSize, fileData, tensorParameterSize, tensorParameter_Dtype, tensorParameter_Offsets);

	fclose(fp);fclose(fr);cJSON_Delete(tensorData);
	assert(munmap(fileData, fileSize) != -1);
}


