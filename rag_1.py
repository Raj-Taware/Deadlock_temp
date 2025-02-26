import networkx as nx
import matplotlib.pyplot as plt

class ResourceAllocationGraph:
    def __init__(self):
        self.G = nx.MultiDiGraph()  # Changed to MultiDiGraph to allow multiple edge types
        self.processes = set()
        self.resources = set()
        
    def add_process(self, process):
        self.G.add_node(process, type='process')
        self.processes.add(process)
        
    def add_resource(self, resource):
        self.G.add_node(resource, type='resource')
        self.resources.add(resource)
        
    def add_assignment_edge(self, resource, process):
        if resource in self.resources and process in self.processes:
            self.G.add_edge(resource, process, type='assignment')
            
    def add_request_edge(self, process, resource):
        if process in self.processes and resource in self.resources:
            self.G.add_edge(process, resource, type='request')
            
    def add_claim_edge(self, process, resource):
        if process in self.processes and resource in self.resources:
            self.G.add_edge(process, resource, type='claim')
            
    def remove_edge(self, src, dest):
        if self.G.has_edge(src, dest):
            self.G.remove_edge(src, dest)
            
    def is_safe(self, process, resource):
        temp_G = self.G.copy()
        temp_G.add_edge(process, resource, type='request')
        try:
            cycle = nx.find_cycle(temp_G, orientation='original')
            if any(temp_G.nodes[node]['type'] == 'resource' for node in cycle):
                return False
            return True
        except nx.NetworkXNoCycle:
            return True

    def detect_deadlock(self):
        try:
            cycle = nx.find_cycle(self.G, orientation='original')
            return cycle
        except nx.NetworkXNoCycle:
            return None

    def visualize(self):
        pos = nx.spring_layout(self.G)
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.processes, node_color='lightblue', node_shape='o', node_size=500)
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.resources, node_color='lightgreen', node_shape='s', node_size=500)
        
        # Draw edges
        assignment_edges = [(u, v) for (u, v, d) in self.G.edges(data=True) if d['type'] == 'assignment']
        request_edges = [(u, v) for (u, v, d) in self.G.edges(data=True) if d['type'] == 'request']
        claim_edges = [(u, v) for (u, v, d) in self.G.edges(data=True) if d['type'] == 'claim']
        
        nx.draw_networkx_edges(self.G, pos, edgelist=assignment_edges, edge_color='b', arrows=True)
        nx.draw_networkx_edges(self.G, pos, edgelist=request_edges, edge_color='r', arrows=True)
        nx.draw_networkx_edges(self.G, pos, edgelist=claim_edges, edge_color='g', style='dashed', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(self.G, pos)
        
        # Add legend
        plt.legend(['Processes', 'Resources', 'Assignment', 'Request', 'Claim'], loc='upper left')
        
        plt.title("Resource Allocation Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Scenario 1: Deadlock via Claim Edges
rag1 = ResourceAllocationGraph()
rag1.add_process('P1'); rag1.add_process('P2')
rag1.add_resource('R1'); rag1.add_resource('R2')

rag1.add_assignment_edge('R1', 'P1')
rag1.add_assignment_edge('R2', 'P2')
rag1.add_claim_edge('P1', 'R2')
rag1.add_claim_edge('P2', 'R1')

print("Scenario 1 (Deadlock):")
print("Deadlock detected:", rag1.detect_deadlock())
rag1.visualize()

# Scenario 3
rag3 = ResourceAllocationGraph()
rag3.add_process('P1');rag3.add_process('P2');rag3.add_process('P3')
for i in range (1,13):
    rag3.add_resource('R'+str(i))
    i=i+1
rag3.add_assignment_edge('R1', 'P1')
rag3.add_assignment_edge('R2', 'P1')
rag3.add_assignment_edge('R3', 'P1')
rag3.add_assignment_edge('R4', 'P1')
rag3.add_assignment_edge('R5', 'P1')
rag3.add_assignment_edge('R6', 'P2')
rag3.add_assignment_edge('R7', 'P2')
rag3.add_assignment_edge('R8', 'P2')
rag3.add_assignment_edge('R9', 'P3')
rag3.add_assignment_edge('R10', 'P3')
rag3.add_claim_edge('P1','R8')
rag3.add_claim_edge('P1','R9')
rag3.add_claim_edge('P1','R10')
rag3.add_claim_edge('P1','R11')
rag3.add_claim_edge('P1','R12')
rag3.add_claim_edge('P2','R11')
rag3.add_claim_edge('P2','R12')
rag3.add_claim_edge('P3','R6')
rag3.add_claim_edge('P3','R7')
rag3.add_claim_edge('P3','R1')
rag3.add_claim_edge('P3','R9')
rag3.add_claim_edge('P3','R10')
rag3.add_claim_edge('P3','R11')
rag3.add_claim_edge('P3','R12')

rag3.visualize()
print("Deadlock detected:", rag3.detect_deadlock())
