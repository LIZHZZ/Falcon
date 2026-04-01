//
//GPU PostOffice - CPU
// 

#include "post_office_solver.cuh"
#include "BitStream/BitWriter.cuh"
#include <climits>

//translated comment
__device__ void calTotalCountAndNonZerosCounts_GPU(
    const int *distribution,
    int *out_pre_non_zeros_count,
    int *out_post_non_zeros_count,
    int *out_total_count,
    int *out_non_zeros_count
) {
    int non_zeros_count = 64; //translated comment
    int total_count = distribution[0];
    out_pre_non_zeros_count[0] = 1; //translated comment
    
    for (int i = 1; i < 64; ++i) {
        total_count += distribution[i];
        //CPU "magic code"
        int magic_code = (distribution[i] == 0) ? 1 : 0;
        non_zeros_count -= magic_code;
        out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1] + (1 - magic_code);
    }
    
    //translated comment
    for (int i = 0; i < 64; ++i) {
        out_post_non_zeros_count[i] = non_zeros_count - out_pre_non_zeros_count[i];
    }
    
    *out_total_count = total_count;
    *out_non_zeros_count = non_zeros_count;
}

//- BuildPostOffice GPU
__device__ int buildPostOffice_GPU(
    const int *distribution,
    int num,
    int non_zeros_count,
    const int *pre_non_zeros_count,
    const int *post_non_zeros_count,
    int *out_positions
) {
    int original_num = num;
    num = min(num, non_zeros_count);
    
    if (num <= 0) {
        return INT_MAX;
    }
    
    //translated comment
    int dp[64][32]; //dp[i][j] = i j
    int pre[64][32]; //pre[i][j] = dp[i][j]
    
    //translated comment
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 32; j++) {
            dp[i][j] = INT_MAX;
            pre[i][j] = -1;
        }
    }
    
    //translated comment
    dp[0][0] = 0;
    pre[0][0] = -1;
    
    //translated comment
    for (int i = 1; i < 64; ++i) {
        if (distribution[i] == 0) {
            continue; //translated comment
        }
        
        for (int j = max(1, num + i - 64); j <= i && j < num; ++j) {
            if (j == 1 && i > 1) {
                //translated comment
                dp[i][j] = 0;
                for (int k = 1; k < i; k++) {
                    dp[i][j] += distribution[k] * k;
                }
                pre[i][j] = 0;
            } else {
                //translated comment
                if (pre_non_zeros_count[i] < j + 1 || 
                    post_non_zeros_count[i] < num - 1 - j) {
                    continue; //translated comment
                }
                
                int app_cost = INT_MAX;
                int pre_k = 0;
                
                for (int k = j - 1; k <= i - 1; ++k) {
                    //translated comment
                    if ((distribution[k] == 0 && k > 0) || 
                        pre_non_zeros_count[k] < j || 
                        post_non_zeros_count[k] < num - j ||
                        dp[k][j-1] == INT_MAX) {
                        continue;
                    }
                    
                    int sum = dp[k][j - 1];
                    for (int p = k + 1; p <= i - 1; ++p) {
                        sum += distribution[p] * (p - k);
                    }
                    
                    if (app_cost > sum) {
                        app_cost = sum;
                        pre_k = k;
                        if (sum == 0) {
                            break; //translated comment
                        }
                    }
                }
                
                if (app_cost != INT_MAX) {
                    dp[i][j] = app_cost;
                    pre[i][j] = pre_k;
                }
            }
        }
    }
    
    //translated comment
    int temp_total_app_cost = INT_MAX;
    int temp_best_last = INT_MAX;
    
    for (int i = num - 1; i < 64; ++i) {
        if (num - 1 == 0 && i > 0) {
            break;
        }
        if ((distribution[i] == 0 && i > 0) || 
            pre_non_zeros_count[i] < num ||
            dp[i][num-1] == INT_MAX) {
            continue;
        }
        
        int sum = dp[i][num - 1];
        for (int j = i + 1; j < 64; ++j) {
            sum += distribution[j] * (j - i);
        }
        
        if (temp_total_app_cost > sum) {
            temp_total_app_cost = sum;
            temp_best_last = i;
        }
    }
    
    if (temp_best_last == INT_MAX) {
        return INT_MAX; //translated comment
    }
    
    //translated comment
    int temp_positions[32];
    int pos_count = 0;
    int current = temp_best_last;
    int j = num - 1;
    
    while (current != -1 && pos_count < 32) {
        temp_positions[pos_count++] = current;
        if (j == 0) break;
        current = pre[current][j];
        j--;
    }
    
    //translated comment
    for (int i = 0; i < pos_count; i++) {
        out_positions[i] = temp_positions[pos_count - 1 - i];
    }
    
    //translated comment
    if (original_num > non_zeros_count) {
        int modifying_positions[32];
        int mod_count = 0;
        int orig_idx = 0, pos_idx = 0;
        
        while (mod_count < original_num && mod_count < 32) {
            if (orig_idx - pos_idx < original_num - pos_count && 
                pos_idx < pos_count && 
                orig_idx < out_positions[pos_idx]) {
                modifying_positions[mod_count++] = orig_idx;
                orig_idx++;
            } else if (pos_idx < pos_count) {
                modifying_positions[mod_count++] = out_positions[pos_idx];
                orig_idx++;
                pos_idx++;
            } else {
                modifying_positions[mod_count++] = orig_idx;
                orig_idx++;
            }
        }
        
        //translated comment
        for (int i = 0; i < mod_count; i++) {
            out_positions[i] = modifying_positions[i];
        }
        pos_count = mod_count;
    }
    
    return pos_count;
}

//- CPU
__device__ int initRoundAndRepresentation(
    const int *distribution,
    int *representation,
    int *round,
    int *out_positions
) {
    //translated comment
    int pre_non_zeros_count[64];
    int post_non_zeros_count[64];
    int total_count, non_zeros_count;
    
    calTotalCountAndNonZerosCounts_GPU(
        distribution, pre_non_zeros_count, post_non_zeros_count,
        &total_count, &non_zeros_count
    );
    
    //translated comment
    int max_z = min(kPositionLength2Bits[non_zeros_count], 5); //5 bit
    int total_cost = INT_MAX;
    int best_positions[32];
    int best_positions_count = 0;
    
    for (int z = 0; z <= max_z; ++z) {
        int present_cost = total_count * z;
        if (present_cost >= total_cost) break; //translated comment
        
        int num = kPow2z[z]; //translated comment
        int temp_positions[32];
        
        int pos_count = buildPostOffice_GPU(
            distribution, num, non_zeros_count,
            pre_non_zeros_count, post_non_zeros_count,
            temp_positions
        );
        
        if (pos_count > 0 && pos_count <= 32) {
            //translated comment
            int app_cost = 0;
            
            //translated comment
            for (int i = 0; i < 64; i++) {
                if (distribution[i] > 0) {
                    int min_dist = INT_MAX;
                    for (int j = 0; j < pos_count; j++) {
                        int dist = abs(i - temp_positions[j]);
                        if (dist < min_dist) {
                            min_dist = dist;
                        }
                    }
                    app_cost += distribution[i] * min_dist;
                }
            }
            
            int temp_total_cost = app_cost + present_cost;
            if (temp_total_cost < total_cost) {
                total_cost = temp_total_cost;
                best_positions_count = pos_count;
                for (int i = 0; i < pos_count; i++) {
                    best_positions[i] = temp_positions[i];
                }
            }
        }
    }
    
    //3. representation round
    for (int i = 0; i < 64; i++) {
        representation[i] = 0;
        round[i] = 0;
    }
    
    if (best_positions_count > 0) {
        representation[0] = 0;
        round[0] = 0;
        int pos_idx = 1;
        
        for (int j = 1; j < 64; ++j) {
            //CPU "magic code"
            int magic_code = (pos_idx < best_positions_count && j == best_positions[pos_idx]) ? 1 : 0;
            representation[j] = representation[j - 1] + magic_code;
            round[j] = magic_code ? j : round[j - 1];
            pos_idx += magic_code;
        }
        
        //translated comment
        for (int i = 0; i < best_positions_count; i++) {
            out_positions[i] = best_positions[i];
        }
    } else {
        //translated comment
        out_positions[0] = 0;
        best_positions_count = 1;
    }
    
    return best_positions_count;
}

__device__ int write_positions_device(
    BitWriter *writer,
    const int *positions,
    int positions_len
) {
    //translated comment
    if (positions_len < 0) positions_len = 0;
    if (positions_len > 32) positions_len = 32;
    
    //translated comment
    write(writer, positions_len, 5);
    int total_bits = 5;
    
    //（6 each）
    for (int i = 0; i < positions_len; i++) {
        int pos = positions[i];
        if (pos < 0) pos = 0;
        if (pos > 63) pos = 63;
        
        write(writer, pos, 6);
        total_bits += 6;
    }
    
    return total_bits;
}

//translated comment
__device__ bool validate_distribution(const int *distribution) {
    int total = 0;
    int non_zero_count = 0;
    
    for (int i = 0; i < 64; i++) {
        if (distribution[i] < 0) return false;
        if (distribution[i] > 1000000) return false;
        
        total += distribution[i];
        if (distribution[i] > 0) {
            non_zero_count++;
        }
    }
    
    return (total <= 10000000 && non_zero_count <= 64);
}

//translated comment
__device__ void debug_print_post_office_result(
    const int *positions, int count, int thread_id
) {
    if (thread_id == 0 && count > 0) {
        printf("PostOffice结果: %d个邮局, 位置: ", count);
        for (int i = 0; i < min(count, 10); i++) {
            printf("%d ", positions[i]);
        }
        printf("\n");
    }
}