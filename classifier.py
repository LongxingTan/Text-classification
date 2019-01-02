class Config:
    def __init__(self,data_dir,bert_config_file,task_name,vocab_file,output_dir,max_seq_length,do_train,do_eval,
                 train_batch_size,eval_batch_size,learning_rate,num_train_epochs,warmup_proportion,save_checkpoints_steps,
                 no_cuda,local_rank,seed,gradient_accumulation_steps,optimize_on_cpu,fp16,loss_scale):
        self.data_dir=data_dir
        self.bert_config_file=bert_config_file
        self.task_name='online'
        self.vocab_file=vocab_file
        self.output_dir=output_dir
        self.init_checkpoint=None
        self.do_lower_case=False
        self.max_seq_length=128
        self.do_train=False
        self.do_eval=False
        self.train_batch_size=32
        self.eval_batch_size=8
        self.learning_rate=5e-5
        self.num_train_epochs=3
        self.warmup_proportion=0.1
        self.save_checkpoints_steps=1000
        self.no_cuda=False
        self.local_rank=-1
        self.seed=42
        self.gradient_accumulation_steps=1
        self.optimize_on_cpu=False
        self.fp16=False
        self.loss_scale=128




def main():
    config = Config()
    processors={
        "cola":ColaProcessor,
        "mnli":MnliProcessor,
        "mrpc":MrpcProcessor,
        "online": OnlineProcessor,
        "cac":Cacprocessor
    }

    if config.local_rank==-1 or config.no_cuda:
        pass

    if config.gradient_accumulation_steps<1:
        raise ValueError("Invalid gradient_accumulation_steps")
    config.train_batch_size=int(config.train_batch_size/config.gradient_accumulation_steps)
    random.seed(config.seed)
    np.random.seed(config.seed)

    bert_config=BertConfig.from_json_file(config.bert_config_file)


    if config.max_seq_length>bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length")

    os.makedirs(config.output_dir,exist_ok=True)
    task_name=config.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found")

    processor=processors[task_name]()


