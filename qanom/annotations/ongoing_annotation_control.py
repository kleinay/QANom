import json


def get_worker_stats(label):
    x = json.load(open(f"data/example/{label}/live/generationWorkerAgreementStats.txt"))
    workers = x.keys()
    def accuracy_from_judgements(l):
        return sum(j['isValid'] for j in l) / float(len(l))
    for worker in workers:
        l = x[worker]['genAgreementJudgments']
        print(f"\nWorker {worker}:")
        print(f"#agreement Judgements: {len(l)}")
        n_correct = sum(j['isValid'] for j in l)
        print(f"general agreement: {accuracy_from_judgements(l)}")
        isVerball = [j for j in l if j['Question'] == 'IsVerbal']
        print(f"is-verbal agreement: {accuracy_from_judgements(isVerball)}% for {len(isVerball)} predicates")
        argl = [j for j in l if j['Question'] != 'IsVerbal']
        print(f"argument agreement: {accuracy_from_judgements(argl)}% for {len(argl)} generated arguments (answers)")

