//
//  tsp_parallel.h
//  PDSassign3
//
//  Created by Gregory Min on 3/5/18.
//  Copyright © 2018 Gregory Min. All rights reserved.
//

#ifndef tsp_parallel_h
#define tsp_parallel_h
#define CITY 64
#define job_tag     100
#define finish_tag  101
#define update_tag  102

#include <stdio.h>

// structure that used to implement a stack as job queue
// data contains visit array, route array, city id, level and cost
struct Node {
    int data[CITY * 2 + 3];
    
    struct Node * next;
    struct Node * end;
};

// get the distance from city a to b
int get_dis(int city_a, int city_b);

// read distance matrix from a file
void load_matrix(const char * name);

// master node update current bound
void master_update(int worlds);

// update worker current bound
void worker_update(void);

// display result: best route
void show_result(void);

// branch and bound algorithm recursive
void tsp(int current_weight, int level, int v[]);

#endif /* tsp_parallel_h */

