from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

NUM_EPOCH = 10


def predict_test(model, test_loader):
    return model.predict_generator(test_loader)


def train(model, train_loader, val_loader, val_labels, model_save_path):
    epoch_id = 0
    best_loss = 1000

    accuracy_values = []
    loss_values = []

    while True:
        model.fit_generator(train_loader)
        y_pred = model.predict_generator(val_loader)

        loss = log_loss(val_labels, y_pred, eps=1e-6)
        accuracy = accuracy_score(val_labels, y_pred.argmax(axis=-1))

        accuracy_values.append(accuracy)
        loss_values.append(loss)

        print("Epoch {0}. Val accuracy {1}. Val loss {2}".format(epoch_id, accuracy, loss))
        model.scheduler_step(loss, epoch_id)
        if loss < best_loss:
            best_loss = loss
            model.save(model_save_path)

        epoch_id += 1

        if epoch_id == NUM_EPOCH:
            break
    
    # Plotting the accuracy and loss
    epochs = range(1, NUM_EPOCH + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy_values, 'bo-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_values, 'ro-', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('accuracy_loss_plot.png')

