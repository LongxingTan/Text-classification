class Config:
    def __init__(self):
        self.data_dir='./data'
        self.bert_config_file='/bert'
        self.task_name='online'
        self.vocab_file='./'
        self.output_dir='./output'
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

