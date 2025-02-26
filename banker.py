import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional

class BankersAlgorithm:
    """
    Implementation of Dijkstra's Banker's Algorithm for deadlock avoidance.
    
    The Banker's Algorithm is a resource allocation and deadlock avoidance
    algorithm that tests for safety by simulating the allocation of predetermined
    maximum possible amounts of all resources, and then checks if satisfying all
    pending requests is possible without entering an unsafe state.
    """
    
    def __init__(self, num_processes: int, num_resources: int):
        """
        Initialize the Banker's Algorithm with the specified number of processes and resources.
        
        Args:
            num_processes: Number of processes in the system
            num_resources: Number of resource types
        """
        self.num_processes = num_processes
        self.num_resources = num_resources
        
        # Available resources vector (shows available units of each resource type)
        self.available = np.zeros(num_resources, dtype=int)
        
        # Maximum demand matrix (shows maximum resource needs for each process)
        self.max_demand = np.zeros((num_processes, num_resources), dtype=int)
        
        # Allocation matrix (shows current resource allocation to each process)
        self.allocation = np.zeros((num_processes, num_resources), dtype=int)
        
        # Need matrix (shows remaining resource needs for each process)
        self.need = np.zeros((num_processes, num_resources), dtype=int)
        
        # Total resources in the system
        self.total_resources = np.zeros(num_resources, dtype=int)
        
        # Process names (for display purposes)
        self.process_names = [f"P{i}" for i in range(num_processes)]
        
        # Resource names (for display purposes)
        self.resource_names = [f"R{i}" for i in range(num_resources)]
        
        # Safe sequence (if one exists)
        self.safe_sequence = []
        
        # Execution trace for visualization
        self.execution_trace = []
    
    def set_total_resources(self, total_resources: List[int]) -> None:
        """
        Set the total number of units for each resource type in the system.
        
        Args:
            total_resources: List of total units for each resource type
        """
        if len(total_resources) != self.num_resources:
            raise ValueError(f"Expected {self.num_resources} resource values, got {len(total_resources)}")
        
        self.total_resources = np.array(total_resources, dtype=int)
        # Initially, all resources are available
        self.available = np.copy(self.total_resources)
    
    def set_max_demand(self, process_id: int, max_resources: List[int]) -> None:
        """
        Set the maximum resource demand for a specific process.
        
        Args:
            process_id: The ID of the process (0-indexed)
            max_resources: List of maximum units needed for each resource type
        """
        if process_id < 0 or process_id >= self.num_processes:
            raise ValueError(f"Invalid process ID: {process_id}")
        
        if len(max_resources) != self.num_resources:
            raise ValueError(f"Expected {self.num_resources} resource values, got {len(max_resources)}")
        
        # Check if the maximum demand is valid (not greater than total resources)
        for i, demand in enumerate(max_resources):
            if demand > self.total_resources[i]:
                raise ValueError(f"Max demand for resource {i} ({demand}) exceeds total available ({self.total_resources[i]})")
        
        self.max_demand[process_id] = np.array(max_resources, dtype=int)
        # Update the need matrix
        self.need[process_id] = self.max_demand[process_id] - self.allocation[process_id]
    
    def allocate_resources(self, process_id: int, resources: List[int]) -> bool:
        """
        Allocate resources to a process.
        
        Args:
            process_id: The ID of the process (0-indexed)
            resources: List of resource units to allocate
            
        Returns:
            bool: True if allocation was successful, False otherwise
        """
        if process_id < 0 or process_id >= self.num_processes:
            raise ValueError(f"Invalid process ID: {process_id}")
        
        if len(resources) != self.num_resources:
            raise ValueError(f"Expected {self.num_resources} resource values, got {len(resources)}")
        
        resources_array = np.array(resources, dtype=int)
        
        # Check if the allocation is valid
        if np.any(self.allocation[process_id] + resources_array > self.max_demand[process_id]):
            print(f"Error: Allocation for P{process_id} would exceed maximum demand")
            return False
        
        if np.any(resources_array > self.available):
            print(f"Error: Not enough available resources for P{process_id}")
            return False
        
        # Update allocation
        self.allocation[process_id] += resources_array
        # Update available resources
        self.available -= resources_array
        # Update need
        self.need[process_id] = self.max_demand[process_id] - self.allocation[process_id]
        
        return True
    
    def release_resources(self, process_id: int, resources: List[int]) -> bool:
        """
        Release resources currently allocated to a process.
        
        Args:
            process_id: The ID of the process (0-indexed)
            resources: List of resource units to release
            
        Returns:
            bool: True if release was successful, False otherwise
        """
        if process_id < 0 or process_id >= self.num_processes:
            raise ValueError(f"Invalid process ID: {process_id}")
        
        if len(resources) != self.num_resources:
            raise ValueError(f"Expected {self.num_resources} resource values, got {len(resources)}")
        
        resources_array = np.array(resources, dtype=int)
        
        # Check if the release is valid
        if np.any(resources_array > self.allocation[process_id]):
            print(f"Error: P{process_id} cannot release more resources than allocated")
            return False
        
        # Update allocation
        self.allocation[process_id] -= resources_array
        # Update available resources
        self.available += resources_array
        # Update need
        self.need[process_id] = self.max_demand[process_id] - self.allocation[process_id]
        
        return True
    
    def safety_algorithm(self, verbose: bool = True) -> Tuple[bool, List[int]]:
        """
        Implements the Safety Algorithm to determine if the system is in a safe state.
        
        The Safety Algorithm works as follows:
        1. Find a process whose needs can be satisfied with the available resources
        2. Add that process to the safe sequence
        3. Assume the process finishes and releases its resources
        4. Repeat until all processes are in the safe sequence or no process can be found
        
        Args:
            verbose: Whether to print detailed steps of the algorithm
            
        Returns:
            Tuple[bool, List[int]]: (is_safe, safe_sequence)
        """
        if verbose:
            print("\n--- SAFETY ALGORITHM EXECUTION ---")
        
        # Make working copies of the arrays
        work = np.copy(self.available)  # Working copy of available resources
        finish = np.zeros(self.num_processes, dtype=bool)  # Track which processes can finish
        
        # Clear previous safe sequence
        self.safe_sequence = []
        self.execution_trace = []
        
        if verbose:
            print(f"Initial work vector: {work}")
            print("Finding a safe execution sequence...")
        
        # Record initial state
        self.execution_trace.append({
            'step': 0,
            'work': np.copy(work),
            'finish': np.copy(finish),
            'message': "Initial state"
        })
        
        # Main loop of the safety algorithm
        step = 1
        while True:
            found = False
            
            # Try to find a process that can be satisfied
            for i in range(self.num_processes):
                trace_message = ""
                
                # Look for a process that's not finished and whose needs can be met
                if not finish[i] and np.all(self.need[i] <= work):
                    if verbose:
                        print(f"Step {step}: Process P{i} can be satisfied with available resources")
                        print(f"  Need: {self.need[i]}")
                        print(f"  Work: {work}")
                    
                    # Mark the process as finished
                    finish[i] = True
                    
                    # Add process to the safe sequence
                    self.safe_sequence.append(i)
                    
                    # Release the resources it was holding
                    work += self.allocation[i]
                    
                    trace_message = f"Process P{i} can finish, releasing resources: {self.allocation[i]}"
                    
                    if verbose:
                        print(f"  P{i} finishes and releases resources: {self.allocation[i]}")
                        print(f"  New work vector: {work}")
                    
                    found = True
                    
                    # Record this step
                    self.execution_trace.append({
                        'step': step,
                        'work': np.copy(work),
                        'finish': np.copy(finish),
                        'process': i,
                        'message': trace_message
                    })
                    
                    step += 1
                    break
            
            if not found:
                # No process could be found, check if we're done or deadlocked
                if np.all(finish):
                    if verbose:
                        print("Safety algorithm complete: System is in a SAFE state")
                        print(f"Safe execution sequence: {' -> '.join([f'P{i}' for i in self.safe_sequence])}")
                    
                    self.execution_trace.append({
                        'step': step,
                        'work': np.copy(work),
                        'finish': np.copy(finish),
                        'message': "All processes can finish - system is SAFE"
                    })
                    
                    return True, self.safe_sequence
                else:
                    # Some processes could not finish - system is unsafe
                    unfinished = [i for i, f in enumerate(finish) if not f]
                    
                    if verbose:
                        print("Safety algorithm complete: System is in an UNSAFE state")
                        print(f"Processes that cannot complete: {', '.join([f'P{i}' for i in unfinished])}")
                    
                    self.execution_trace.append({
                        'step': step,
                        'work': np.copy(work),
                        'finish': np.copy(finish),
                        'message': f"Cannot proceed - processes {unfinished} cannot finish"
                    })
                    
                    return False, []
    
    def resource_request_algorithm(self, process_id: int, request: List[int], verbose: bool = True) -> bool:
        """
        Implements the Resource Request Algorithm to determine if a resource request
        can be granted immediately.
        
        The Resource Request Algorithm works as follows:
        1. Check if the request exceeds the process's maximum claim
        2. Check if the request exceeds available resources
        3. Pretend to allocate the resources and check if the resulting state is safe
        4. If safe, grant the request; otherwise, deny it
        
        Args:
            process_id: The ID of the requesting process (0-indexed)
            request: List of requested resource units
            verbose: Whether to print detailed steps of the algorithm
            
        Returns:
            bool: True if the request is granted, False otherwise
        """
        if verbose:
            print(f"\n--- RESOURCE REQUEST ALGORITHM FOR P{process_id} ---")
            print(f"Request vector: {request}")
        
        request_array = np.array(request, dtype=int)
        
        # Step 1: Check if request exceeds maximum claim
        if np.any(request_array > self.max_demand[process_id]):
            if verbose:
                print("Request denied: Exceeds maximum claim")
                print(f"  Request: {request_array}")
                print(f"  Max demand: {self.max_demand[process_id]}")
            return False
        
        # Step 2: Check if request exceeds need
        if np.any(request_array > self.need[process_id]):
            if verbose:
                print("Request denied: Exceeds current need")
                print(f"  Request: {request_array}")
                print(f"  Need: {self.need[process_id]}")
            return False
        
        # Step 3: Check if request exceeds available resources
        if np.any(request_array > self.available):
            if verbose:
                print("Request denied: Exceeds available resources")
                print(f"  Request: {request_array}")
                print(f"  Available: {self.available}")
            return False
        
        if verbose:
            print("Initial checks passed - testing if resulting state is safe...")
        
        # Step 4: Pretend to allocate resources and check for safety
        
        # Save current state
        old_available = np.copy(self.available)
        old_allocation = np.copy(self.allocation)
        old_need = np.copy(self.need)
        
        # Pretend to allocate
        self.available -= request_array
        self.allocation[process_id] += request_array
        self.need[process_id] -= request_array
        
        # Check if resulting state is safe
        is_safe, sequence = self.safety_algorithm(verbose=False)
        
        if is_safe:
            if verbose:
                print("Request granted: Resulting state is safe")
                print(f"  Safe sequence: {' -> '.join([f'P{i}' for i in sequence])}")
            return True
        else:
            # Restore old state
            self.available = old_available
            self.allocation = old_allocation
            self.need = old_need
            
            if verbose:
                print("Request denied: Resulting state would be unsafe")
            return False
    
    def display_state(self) -> None:
        """
        Display the current state of the system in a formatted table.
        """
        print("\n===== CURRENT SYSTEM STATE =====")
        
        # Display resources
        print("\nTotal Resources:")
        for i, total in enumerate(self.total_resources):
            print(f"  {self.resource_names[i]}: {total}")
        
        print("\nAvailable Resources:")
        for i, avail in enumerate(self.available):
            print(f"  {self.resource_names[i]}: {avail}")
        
        # Create a dataframe for better display
        data = []
        for i in range(self.num_processes):
            row = {
                'Process': self.process_names[i],
                'Allocation': str(self.allocation[i]),
                'Max Demand': str(self.max_demand[i]),
                'Need': str(self.need[i])
            }
            data.append(row)
        
        # Convert to DataFrame and print
        df = pd.DataFrame(data)
        print("\nProcess Details:")
        print(df.to_string(index=False))
        
        # Check if the system is in a safe state
        is_safe, sequence = self.safety_algorithm(verbose=False)
        
        if is_safe:
            print(f"\nSystem is in a SAFE state.")
            print(f"Safe sequence: {' -> '.join([f'P{i}' for i in sequence])}")
        else:
            print("\nSystem is in an UNSAFE state. No safe execution sequence exists.")
        
        print("================================\n")
    
    def visualize_safety_algorithm(self, figsize=(12, 8), save_path=None):
        """
        Visualize the steps of the safety algorithm with proper column consistency.
        """
        if not self.execution_trace:
            print("No execution trace available. Run the safety algorithm first.")
            return

        num_steps = len(self.execution_trace)
        fig, axs = plt.subplots(num_steps, 1, figsize=figsize, gridspec_kw={'hspace': 0.5})
        if num_steps == 1:
            axs = [axs]

        for step_idx, step_data in enumerate(self.execution_trace):
            ax = axs[step_idx]
            ax.set_title(f"Step {step_data['step']}: {step_data.get('message', '')}", pad=20)
            
            # Build table content with consistent columns (1 label + num_resources)
            cell_text = []
            col_labels = ['State'] + self.resource_names
            
            # 1. Add work vector (available resources)
            work_row = ['Work'] + [str(w) for w in step_data['work']]
            cell_text.append(work_row)
            
            # 2. Add process allocations and needs
            for p in range(self.num_processes):
                alloc_row = [f"P{p} Alloc"] + [str(a) for a in self.allocation[p]]
                need_row = [f"P{p} Need"] + [str(n) for n in self.need[p]]
                cell_text.append(alloc_row)
                cell_text.append(need_row)

            # Create table
            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc='center',
                cellLoc='center',
                colWidths=[0.15] + [0.1]*self.num_resources
            )
            
            # 3. Highlight finished processes using the 'finish' array
            for p in range(self.num_processes):
                if step_data['finish'][p]:
                    # Calculate row positions (0=header, 1=work, then 2 rows per process)
                    alloc_row = 1 + (p * 2) + 1  # +1 for header row
                    need_row = alloc_row + 1
                    
                    # Highlight allocation and need rows
                    for col in range(len(col_labels)):
                        table[alloc_row, col].set_facecolor('#90EE90')  # Lightgreen
                        table[need_row, col].set_facecolor('#90EE90')

            # 4. Highlight current process if specified
            if 'process' in step_data:
                p = step_data['process']
                alloc_row = 1 + (p * 2) + 1  # +1 for header row
                need_row = alloc_row + 1
                
                for col in range(len(col_labels)):
                    table[alloc_row, col].set_edgecolor('red')
                    table[alloc_row, col].set_linewidth(1.5)
                    table[need_row, col].set_edgecolor('red')
                    table[need_row, col].set_linewidth(1.5)

            # Formatting
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    
    def visualize_resource_allocation(self, figsize=(10, 6), save_path=None):
        """
        Visualize the current resource allocation state.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Resource allocation by process
        process_names = [f"P{i}" for i in range(self.num_processes)]
        resource_names = [f"R{i}" for i in range(self.num_resources)]
        
        # Create a stacked bar chart for resource allocation
        bottom = np.zeros(self.num_resources)
        
        for i, process in enumerate(process_names):
            ax1.bar(resource_names, self.allocation[i], bottom=bottom, label=process)
            bottom += self.allocation[i]
        
        # Add available resources on top
        ax1.bar(resource_names, self.available, bottom=bottom, label="Available",
               color='lightgray', edgecolor='gray')
        
        ax1.set_title("Resource Allocation by Process")
        ax1.set_xlabel("Resource Type")
        ax1.set_ylabel("Units")
        ax1.legend()
        
        # Need vs Max visualization
        x = np.arange(len(process_names))
        width = 0.35
        
        for i, resource in enumerate(resource_names):
            # Needs
            ax2.bar(x - width/2 + (i * width/self.num_resources), 
                   self.need[:, i], width=width/self.num_resources, 
                   label=f"{resource} Need" if i == 0 else "_nolegend_")
            
            # Max demand
            ax2.bar(x + width/2 + (i * width/self.num_resources), 
                   self.max_demand[:, i], width=width/self.num_resources,
                   alpha=0.5, label=f"{resource} Max" if i == 0 else "_nolegend_")
        
        ax2.set_title("Process Needs vs Max Demand")
        ax2.set_xlabel("Process")
        ax2.set_ylabel("Units")
        ax2.set_xticks(x)
        ax2.set_xticklabels(process_names)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

# Example usage function
def run_bankers_algorithm_example_1():
    """
    Run a complete example of the Banker's Algorithm, demonstrating both the
    Safety Algorithm and Resource Request Algorithm.
    """
    print("====== BANKER'S ALGORITHM DEMONSTRATION ======")
    
    # Create a system with 5 processes and 3 resources
    banker = BankersAlgorithm(5, 3)
    
    # Set total resources in the system
    banker.set_total_resources([10, 5, 7])
    
    # Set maximum demands for each process
    banker.set_max_demand(0, [7, 5, 3])  # P0 needs at most 7 of R0, 5 of R1, 3 of R2
    banker.set_max_demand(1, [3, 2, 2])  # P1 needs at most 3 of R0, 2 of R1, 2 of R2
    banker.set_max_demand(2, [9, 0, 2])  # P2 needs at most 9 of R0, 0 of R1, 2 of R2
    banker.set_max_demand(3, [2, 2, 2])  # P3 needs at most 2 of R0, 2 of R1, 2 of R2
    banker.set_max_demand(4, [4, 3, 3])  # P4 needs at most 4 of R0, 3 of R1, 3 of R2
    
    # Initial allocation (this allocation should be safe)
    print("\n--- INITIAL RESOURCE ALLOCATION ---")
    banker.allocate_resources(0, [0, 1, 0])  # P0 has 0 of R0, 1 of R1, 0 of R2
    banker.allocate_resources(1, [2, 0, 0])  # P1 has 2 of R0, 0 of R1, 0 of R2
    banker.allocate_resources(2, [3, 0, 2])  # P2 has 3 of R0, 0 of R1, 2 of R2
    banker.allocate_resources(3, [2, 1, 1])  # P3 has 2 of R0, 1 of R1, 1 of R2
    banker.allocate_resources(4, [0, 0, 2])  # P4 has 0 of R0, 0 of R1, 2 of R2
    
    # Display the current state
    banker.display_state()
    
    # Visualize the resource allocation
    banker.visualize_resource_allocation()
    
    # Run the safety algorithm
    is_safe, sequence = banker.safety_algorithm()
    
    # Visualize the safety algorithm steps
    banker.visualize_safety_algorithm()
    
    # Try a safe resource request
    print("\n--- TESTING SAFE RESOURCE REQUEST ---")
    print("P1 requests [1, 0, 2]")
    result = banker.resource_request_algorithm(1, [1, 0, 2])
    
    if result:
        print("Request was granted!")
        banker.allocate_resources(1, [1, 0, 2])
        banker.display_state()
    else:
        print("Request was denied.")
    
    # Try an unsafe resource request
    print("\n--- TESTING UNSAFE RESOURCE REQUEST ---")
    print("P4 requests [3, 3, 0] (which would make the system unsafe)")
    result = banker.resource_request_algorithm(4, [3, 3, 0])
    
    if result:
        print("Request was granted!")
        banker.allocate_resources(4, [3, 3, 0])
    else:
        print("Request was denied, as expected.")
    
    # Display the final state
    banker.display_state()
    
    # Demonstrate gradual deadlock formation
    print("\n--- DEMONSTRATING UNSAFE STATE FORMATION ---")
    # Make allocations that lead toward an unsafe state
    banker.allocate_resources(0, [4, 2, 0])  # P0 gets more resources
    banker.display_state()
    
    # Run the safety algorithm again
    is_safe, sequence = banker.safety_algorithm()
    
    # Visualize the final safety algorithm steps
    #banker.visualize_safety_algorithm()
    
    return banker



def run_bankers_algorithm_example_2():
    """
    Run a complete example of the Banker's Algorithm, demonstrating both the
    Safety Algorithm and Resource Request Algorithm.
    """
    print("====== BANKER'S ALGORITHM DEMONSTRATION ======")
    
    # Create a system with 5 processes and 3 resources
    banker = BankersAlgorithm(3, 3)
    
    # Set total resources in the system
    banker.set_total_resources([2, 2, 2])
    
    # Set maximum demands for each process
    banker.set_max_demand(0, [2,1,1])  # P0 needs at most 7 of R0, 5 of R1, 3 of R2
    banker.set_max_demand(1, [1,2,1])  # P1 needs at most 3 of R0, 2 of R1, 2 of R2
    banker.set_max_demand(2, [1,1,2])  # P2 needs at most 9 of R0, 0 of R1, 2 of R2

    
    # Initial allocation (this allocation should be safe)
    print("\n--- INITIAL RESOURCE ALLOCATION ---")
    banker.allocate_resources(0, [2, 0, 0])  # P0 has 0 of R0, 1 of R1, 0 of R2
    banker.allocate_resources(1, [0, 2, 0])  # P1 has 2 of R0, 0 of R1, 0 of R2
    banker.allocate_resources(2, [0, 0, 2])  # P2 has 3 of R0, 0 of R1, 2 of R2

    
    # Display the current state
    banker.display_state()
    
    # Visualize the resource allocation
    banker.visualize_resource_allocation()
    
    # Run the safety algorithm
    is_safe, sequence = banker.safety_algorithm()
    
    # Visualize the safety algorithm steps
    banker.visualize_safety_algorithm()
    
    # Try a safe resource request
    print("\n--- TESTING SAFE RESOURCE REQUEST ---")
    print("P1 requests [1, 0, 2]")
    result = banker.resource_request_algorithm(1, [0, 1, 1])
    
    if result:
        print("Request was granted!")
        banker.allocate_resources(1, [1, 0, 2])
        banker.display_state()
    else:
        print("Request was denied.")

    # Display the final state
    banker.display_state()
    
    banker.display_state()
    
    # Run the safety algorithm again
    is_safe, sequence = banker.safety_algorithm()
    
    # Visualize the final safety algorithm steps
    #banker.visualize_safety_algorithm()
    
    return banker

if __name__ == "__main__":
    run_bankers_algorithm_example_2()