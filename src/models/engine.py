import torch


class Engine:
    def __init__(self, model, optimizer, scheduler, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device

    def train_one_step(self, data):
        inputs, labels = data[0].to(self.device), data[1].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        for data in data_loader:
            loss = self.train_one_step(data)
            self.scheduler.step()
            total_loss += loss
        return total_loss / len(data_loader)

    def validate_one_step(self, data):
        inputs, labels = data[0].to(self.device), data[1].to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        outputs = torch.argmax(outputs, axis=1)
        return (
            loss.detach().cpu(),
            labels.detach().cpu().numpy().tolist(),
            outputs.detach().cpu().numpy().tolist(),
        )

    def validate_one_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0
        final_labels, final_outputs = [], []
        for data in data_loader:
            with torch.no_grad():
                loss, label, output = self.validate_one_step(data)
            total_loss += loss
            final_labels.extend(label)
            final_outputs.extend(output)
        return total_loss / len(data_loader), final_labels, final_outputs

