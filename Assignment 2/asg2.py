import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import random
import warnings
warnings.filterwarnings('ignore')

def load_facebook_graph():
    print("Loading Facebook social network...")
    
    try:
        G = nx.read_edgelist('facebook_combined.txt', nodetype=int)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except:
        print("Creating sample graph (download facebook_combined.txt for real data)")
        G = nx.barabasi_albert_graph(1000, 5)
    
    return G

def extract_graph_features(G):
    print("Extracting graph features...")
    
    features = {}
    
    degree_cent = nx.degree_centrality(G)
    
    if G.number_of_nodes() > 1000:
        betweenness = nx.betweenness_centrality(G, k=100)
    else:
        betweenness = nx.betweenness_centrality(G)
    
    clustering = nx.clustering(G)
    
    pagerank = nx.pagerank(G)
    
    for node in G.nodes():
        features[node] = {
            'degree': G.degree(node),
            'degree_centrality': degree_cent[node],
            'betweenness_centrality': betweenness.get(node, 0),
            'clustering_coefficient': clustering[node],
            'pagerank': pagerank[node],
            'neighbors': len(list(G.neighbors(node)))
        }
    
    return pd.DataFrame.from_dict(features, orient='index')

def print_graph_metrics(G):
    print("\n" + "="*60)
    print("GRAPH METRICS")
    print("="*60)
    
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    
    degree_cent = nx.degree_centrality(G)
    avg_degree_cent = sum(degree_cent.values()) / len(degree_cent)
    
    if G.number_of_nodes() > 1000:
        betweenness = nx.betweenness_centrality(G, k=100)
    else:
        betweenness = nx.betweenness_centrality(G)
    avg_betweenness = sum(betweenness.values()) / len(betweenness)
    
    clustering = nx.clustering(G)
    avg_clustering = sum(clustering.values()) / len(clustering)
    
    pagerank = nx.pagerank(G)
    avg_pagerank = sum(pagerank.values()) / len(pagerank)
    
    print(f"Average Degree: {avg_degree:.4f}")
    print(f"Average Degree Centrality: {avg_degree_cent:.4f}")
    print(f"Average Betweenness Centrality: {avg_betweenness:.4f}")
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
    print(f"Average PageRank: {avg_pagerank:.4f}")
    print("="*60)

def create_bot_labels(G, bot_ratio=0.1):
    print(f"Creating bot labels ({bot_ratio*100}% bots)...")
    
    labels = {}
    nodes = list(G.nodes())
    
    degrees = dict(G.degree())
    sorted_nodes = sorted(nodes, key=lambda x: degrees[x])
    
    num_bots = int(len(nodes) * bot_ratio)
    bot_nodes = set(sorted_nodes[:num_bots])
    
    for node in nodes:
        labels[node] = 1 if node in bot_nodes else 0
    
    return labels, bot_nodes

def train_detector(features_df, labels):
    print("Training bot detector...")
    
    X = features_df.values
    y = np.array([labels[i] for i in features_df.index])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'model': clf,
        'test_indices': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return results

def structural_evasion_attack(G, bot_nodes, num_edges=50):
    print("Performing Structural Evasion Attack...")
    
    G_attack = G.copy()
    
    degrees = dict(G_attack.degree())
    legitimate_nodes = [n for n in G_attack.nodes() if n not in bot_nodes]
    high_degree_nodes = sorted(legitimate_nodes, key=lambda x: degrees[x], reverse=True)[:50]
    
    edges_added = 0
    for bot in random.sample(list(bot_nodes), min(len(bot_nodes), num_edges)):
        target = random.choice(high_degree_nodes)
        if not G_attack.has_edge(bot, target):
            G_attack.add_edge(bot, target)
            edges_added += 1
    
    print(f"Added {edges_added} edges in evasion attack")
    return G_attack

def graph_poisoning_attack(G, bot_nodes, num_fake_nodes=30):
    print("Performing Graph Poisoning Attack...")
    
    G_poison = G.copy()
    
    max_node = max(G_poison.nodes()) + 1
    fake_nodes = list(range(max_node, max_node + num_fake_nodes))
    
    for fake_node in fake_nodes:
        G_poison.add_node(fake_node)
        
        bot_targets = random.sample(list(bot_nodes), min(3, len(bot_nodes)))
        for bot in bot_targets:
            G_poison.add_edge(fake_node, bot)
        
        legit_nodes = [n for n in G.nodes() if n not in bot_nodes]
        legit_targets = random.sample(legit_nodes, min(5, len(legit_nodes)))
        for legit in legit_targets:
            G_poison.add_edge(fake_node, legit)
    
    print(f"Added {num_fake_nodes} fake nodes in poisoning attack")
    return G_poison, fake_nodes

def visualize_graph(G, labels, title, filename, bot_nodes=None):
    print(f"Creating visualization: {title}...")
    
    plt.figure(figsize=(12, 8))
    
    if G.number_of_nodes() > 200:
        sampled_nodes = random.sample(list(G.nodes()), 200)
        G_vis = G.subgraph(sampled_nodes)
    else:
        G_vis = G
    
    pos = nx.spring_layout(G_vis, k=0.5, iterations=50)
    
    node_colors = []
    for node in G_vis.nodes():
        if bot_nodes and node in bot_nodes:
            node_colors.append('red')
        elif labels.get(node, 0) == 1:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')
    
    nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors, 
                          node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G_vis, pos, alpha=0.2)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def evaluate_scenario(scenario_name, G, bot_nodes, original_labels):
    print(f"\n{'='*50}")
    print(f"Evaluating: {scenario_name}")
    print(f"{'='*50}")
    
    print_graph_metrics(G)
    
    features_df = extract_graph_features(G)
    
    labels = original_labels.copy()
    for node in G.nodes():
        if node not in labels:
            labels[node] = 0
    
    results = train_detector(features_df, labels)
    
    print(f"\nResults for {scenario_name}:")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    
    return results, features_df

def main():
    print("="*60)
    print("Social Network Bot Detection with Graph Attacks")
    print("="*60)
    
    G_original = load_facebook_graph()
    
    labels, bot_nodes = create_bot_labels(G_original, bot_ratio=0.1)
    
    all_results = {}
    
    print("\n" + "="*60)
    print("BASELINE SCENARIO")
    print("="*60)
    results_baseline, features_baseline = evaluate_scenario(
        "Baseline (No Attack)", G_original, bot_nodes, labels
    )
    all_results['baseline'] = results_baseline
    visualize_graph(G_original, labels, "Baseline Graph", 
                   "baseline_graph.png", bot_nodes)
    
    print("\n" + "="*60)
    print("STRUCTURAL EVASION ATTACK")
    print("="*60)
    G_evasion = structural_evasion_attack(G_original, bot_nodes, num_edges=50)
    results_evasion, features_evasion = evaluate_scenario(
        "After Structural Evasion", G_evasion, bot_nodes, labels
    )
    all_results['evasion'] = results_evasion
    visualize_graph(G_evasion, labels, "After Structural Evasion Attack", 
                   "evasion_graph.png", bot_nodes)
    
    print("\n" + "="*60)
    print("GRAPH POISONING ATTACK")
    print("="*60)
    G_poison, fake_nodes = graph_poisoning_attack(G_original, bot_nodes, num_fake_nodes=30)
    results_poison, features_poison = evaluate_scenario(
        "After Graph Poisoning", G_poison, bot_nodes, labels
    )
    all_results['poisoning'] = results_poison
    visualize_graph(G_poison, labels, "After Graph Poisoning Attack", 
                   "poisoning_graph.png", bot_nodes)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Baseline': [
            all_results['baseline']['accuracy'],
            all_results['baseline']['precision'],
            all_results['baseline']['recall'],
            all_results['baseline']['f1']
        ],
        'Structural Evasion': [
            all_results['evasion']['accuracy'],
            all_results['evasion']['precision'],
            all_results['evasion']['recall'],
            all_results['evasion']['f1']
        ],
        'Graph Poisoning': [
            all_results['poisoning']['accuracy'],
            all_results['poisoning']['precision'],
            all_results['poisoning']['recall'],
            all_results['poisoning']['f1']
        ]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    print("\n" + comparison_df.to_string())
    comparison_df.to_csv('comparison_results.csv')
    
    plt.figure(figsize=(12, 6))
    comparison_df.T.plot(kind='bar', ax=plt.gca())
    plt.title('Bot Detection Performance Comparison', fontsize=14)
    plt.ylabel('Score')
    plt.xlabel('Scenario')
    plt.legend(loc='lower right')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
