#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "bktree.h"

static int write_string(BKTree * bktree, char * string, unsigned char len);
static BKNode * write_new_record(BKTree * bktree, char * string, unsigned char len);

BKTree * bktree_new(int (* distance)(char *, int, char *, int, int)) {
    BKTree * bktree = (BKTree *)malloc(sizeof(BKTree));

    bktree->tree_size = BKTREE_TREE_SIZE;
    bktree->tree = (BKNode *)malloc(bktree->tree_size);
    bktree->tree_cursor = bktree->tree;
    
    bktree->strings_size = BKTREE_STRINGS_SIZE;
    bktree->strings = (char*)malloc(bktree->strings_size);
    bktree->strings_cursor = bktree->strings;

    bktree->size = 0;

    bktree->distance = distance;

    return bktree;
}

void bktree_destroy(BKTree * bktree) {
    free(bktree->tree);
    free(bktree->strings);
    free(bktree);
}

BKNode * bktree_add(BKTree * bktree, char * string, unsigned char len) {
    if(len > BKTREE_STRING_MAX || len == 0)
        return NULL;

    if(bktree->size == 0) {
        return write_new_record(bktree, string, len);
    }

    BKNode * node = (BKNode *) bktree->tree;
    while(node) {
        char * node_str = BKTREE_GET_STRING(bktree, node->string_offset);
        int node_str_len = BKTREE_GET_STRING_LEN(bktree, node->string_offset);

        int d = bktree->distance(node_str, node_str_len, string, len, -1);

        if(d == 0)
            return BKTREE_OK;

        if(node->next[d] > 0) {
            node = bktree->tree + node->next[d];
        } else {
            BKNode * new_node = write_new_record(bktree, string, len);
            node->next[d] = new_node - bktree->tree;
            return new_node;
        }
    }

    return NULL;
}

// BKResult * bktree_result_new(BKResult * next, BKNode * node, int distance) {
//     BKResult * result = (BKResult *)malloc(sizeof(BKResult));
//     result->next = next;
//     result->distance = distance;
//     result->string_offset = node->string_offset;
// 
//     return result;
// }


void inner_query(BKTree * bktree, BKNode * node, char * string, unsigned char len, int max, std::vector<BKResult>& res) {

    int d = bktree->distance(BKTREE_GET_STRING(bktree, node->string_offset), BKTREE_GET_STRING_LEN(bktree, node->string_offset), string, len, -1);

    int start = d - max < 1 ? 1 : d - max;
    int stop = d + max + 1;
    if(stop >= BKTREE_STRING_MAX)
        stop = BKTREE_STRING_MAX - 1;

    if(d <= max) {
       // *result_ptr = bktree_result_new(*result_ptr, node, d);
		BKResult r;
		r.distance = d;
		int len = bktree->strings[node->string_offset];
		char* start = bktree->strings + node->string_offset + 1;
		char* end = start + len;
		r.str = std::string(start,end);
		res.push_back(r);
    }

    int i;
    for(i = start; i <= stop; i++) {
        if(node->next[i] > 0) {
            inner_query(bktree, bktree->tree + node->next[i], string, len, max, res);
        }
    }
}

std::vector<BKResult> bktree_query(BKTree * bktree, char * string, unsigned char len, int max) {
	std::vector<BKResult> res;
    inner_query(bktree, bktree->tree, string, len, max, res);
    return res;
}

void bktree_node_print(BKTree * bktree, BKNode * node) {
    if(bktree == NULL) {
        printf("bktree is null\n");
        return;
    }

    if(node == NULL) {
        printf("node is null\n");
        return;
    }

    printf("String: %s\n", BKTREE_GET_STRING(bktree, node->string_offset));
    printf("Offset: %ld\n", node - bktree->tree);
    int i;
    for(i = 0; i < BKTREE_STRING_MAX; i++)
        printf("%d ", node->next[i]);

    printf("\n");
}

static int write_string(BKTree * bktree, char * string, unsigned char len) {
    while(bktree->strings_cursor - bktree->strings + len + 2 >= bktree->strings_size) {
        int cursor_offset = bktree->strings_cursor - bktree->strings;

        char * old_strings = bktree->strings;
        bktree->strings = (char*)malloc(bktree->strings_size * 2);
        memcpy(bktree->strings, old_strings, bktree->strings_size);
        free(old_strings);

        //printf("old ptr: %p\n", old_strings);
        //printf("new ptr: %p\n", bktree->strings);

        bktree->strings_size *= 2;
        bktree->strings_cursor = bktree->strings + cursor_offset;
    }

    int original_offset = bktree->strings_cursor - bktree->strings;

    *(bktree->strings_cursor) = len;
    memcpy(bktree->strings_cursor + 1, string, len);
    *(bktree->strings_cursor + len + 1) = '\0';
    bktree->strings_cursor += len + 2;

    return original_offset;
}

static BKNode * write_new_record(BKTree * bktree, char * string, unsigned char len) {
    BKNode * node = bktree->tree_cursor++;
    node->string_offset = write_string(bktree, string, len);
    
    int i;
    for(i = 0; i < BKTREE_STRING_MAX; i++)
        node->next[i] = 0;
    
    bktree->size++;
    
    return node;
}