#include <stdio.h>
#include <stdlib.h>

// Represents a free block in our conceptual free list
typedef struct FreeBlock {
    int address;
    int size;
    struct FreeBlock* next;
} FreeBlock;

// Function to initialize a free list with a single block
FreeBlock* initialize_free_list(int total_size) {
    FreeBlock* head = (FreeBlock*)malloc(sizeof(FreeBlock));
    if (head == NULL) {
        perror("Failed to allocate free list head");
        exit(EXIT_FAILURE);
    }
    head->address = 0;
    head->size = total_size;
    head->next = NULL;
    return head;
}

// Function to find a free block large enough for a request (first fit for simplicity)
FreeBlock* find_free_block(FreeBlock* head, int requested_size) {
    FreeBlock* current = head;
    while (current != NULL) {
        if (current->size >= requested_size) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// Function to perform splitting
FreeBlock* split_block(FreeBlock* block, int requested_size) {
    if (block->size > requested_size) {
        // Create a new free block for the remainder
        FreeBlock* remainder = (FreeBlock*)malloc(sizeof(FreeBlock));
        if (remainder == NULL) {
            perror("Failed to allocate remainder block");
            exit(EXIT_FAILURE);
        }
        remainder->address = block->address + requested_size;
        remainder->size = block->size - requested_size;
        remainder->next = block->next;
        block->size = requested_size;
        block->next = remainder; // Insert the remainder after the allocated part
        return block; // Return the (now smaller) allocated block
    }
    return block; // No split needed
}

// Function to perform coalescing (checks the next block only for simplicity)
FreeBlock* coalesce_blocks(FreeBlock* head) {
    FreeBlock* current = head;
    while (current != NULL && current->next != NULL) {
        if (current->address + current->size == current->next->address) {
            // Blocks are adjacent, coalesce them
            current->size += current->next->size;
            FreeBlock* temp = current->next;
            current->next = current->next->next;
            free(temp);
        } else {
            current = current->next;
        }
    }
    return head;
}

// Helper function to print the free list
void print_free_list(FreeBlock* head) {
    printf("Free List: ");
    FreeBlock* current = head;
    while (current != NULL) {
        printf("[%d, size:%d] -> ", current->address, current->size);
        current = current->next;
    }
    printf("NULL\n");
}

int main() {
    // Initialize a heap of 30 bytes, initially all free
    FreeBlock* free_list = initialize_free_list(30);
    printf("Initial ");
    print_free_list(free_list);

    // Request 5 bytes (splitting will occur)
    int request1_size = 5;
    FreeBlock* block1 = find_free_block(free_list, request1_size);
    if (block1 != NULL) {
        printf("Allocating %d bytes at address %d\n", request1_size, block1->address);
        split_block(block1, request1_size);
        printf("After first allocation: ");
        print_free_list(free_list);
    }

    // Request 10 bytes from the remainder (further splitting)
    int request2_size = 10;
    FreeBlock* block2 = find_free_block(free_list->next, request2_size); // Start search from the remainder
    if (block2 != NULL) {
        printf("Allocating %d bytes at address %d\n", request2_size, block2->address);
        split_block(block2, request2_size);
        printf("After second allocation: ");
        print_free_list(free_list);
    }

    // Simulate freeing the block at address 5 (size 5)
    // We need to manually update our free list in this simplified example
    FreeBlock* freed_block = (FreeBlock*)malloc(sizeof(FreeBlock));
    freed_block->address = 5;
    freed_block->size = 5;
    freed_block->next = free_list->next; // Point to the block after the first allocated one
    free_list->next = freed_block;        // Insert the freed block

    printf("After freeing block at address 5: ");
    print_free_list(free_list);

    // Now perform coalescing
    free_list = coalesce_blocks(free_list);
    printf("After first coalescing: ");
    print_free_list(free_list);

    // Simulate freeing the block at address 15 (size 10)
    FreeBlock* freed_block2 = (FreeBlock*)malloc(sizeof(FreeBlock));
    freed_block2->address = 15;
    freed_block2->size = 10;
    freed_block2->next = free_list->next->next; // Point after the second allocated block
    free_list->next->next = freed_block2;        // Insert

    printf("After freeing block at address 15: ");
    print_free_list(free_list);

    // Perform another coalescing (now two adjacent free blocks should merge)
    free_list = coalesce_blocks(free_list);
    printf("After second coalescing: ");
    print_free_list(free_list);

    // Free the allocated memory for the free list nodes (important for a real implementation)
    FreeBlock* current = free_list;
    while (current != NULL) {
        FreeBlock* next = current->next;
        free(current);
        current = next;
    }

    return 0;
}