""" Example for MOGONET classification
"""
from train_test import train_test

if __name__ == "__main__":    
    data_folder = 'BRCA'
    view_list = [1,2,3]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    
    # Eseguire training e ottenere i risultati
    results = train_test(data_folder, view_list, num_class,
                        lr_e_pretrain, lr_e, lr_c, 
                        num_epoch_pretrain, num_epoch)
    
    # Stampare un riepilogo dei risultati finali
    print("\n" + "="*60)
    print("RIEPILOGO FINALE")
    print("="*60)
    print(f"Dataset: {data_folder}")
    print(f"Viste utilizzate: {view_list}")
    print(f"Numero di campioni test: {len(results['test_indices'])}")
    print(f"Modelli salvati in: {data_folder}/models/")
    
    # Calcolo accuracy finale
    from sklearn.metrics import accuracy_score
    final_accuracy = accuracy_score(results['true_labels'], 
                                   results['predictions'].argmax(1))
    print(f"Accuracy finale: {final_accuracy:.4f}")
    print("="*60)             