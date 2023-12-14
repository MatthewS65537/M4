import sys
sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")

from master_init import *

model = INITIALIZE_MODEL(device=device, device_ids=device_ids).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

dsg_tasks = DSGTasks()
dsg_tasks.add_task(
    DSGTask(
        "EEG-TXT-BART",
        dataset=zuco_dataloader,
        converge_lim=2,
        converge_threshold=0.05,
        div_threshold=0.01
        )
    )
dsg_tasks.add_task(
    DSGTask(
        "EEG-IMG-DIFFUSION",
        dataset=brain2image_dataloader,
        converge_lim=2,
        converge_threshold=0.005,
        div_threshold=0.01
        )
    )


