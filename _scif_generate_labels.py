import pandas as pd
import numpy as np

attack_ratios = ['0.01', '0.03', '0.05']

for ratio in attack_ratios:
    path = f'./Data/Process/MovieLens/{ratio}/'

    print(f'===== {ratio} 시작 =====')

    # train_normal과 train_random CSV 파일 읽기
    train_normal_df = pd.read_csv(path + 'train_normal.csv')
    train_random_df = pd.read_csv(path + 'train_random.csv')

    # 새로운 label 저장 배열
    new_labels = np.zeros(len(train_random_df))

    # train_random의 각 행 처리
    for index, row in train_random_df.iterrows():
        u = row['user']
        i = row['item']

        user_records = train_normal_df[
            train_normal_df['user'] == u
        ]

        if len(user_records) > 0:
            avg_label = user_records['label'].mean()
        else:
            item_records = train_normal_df[
                train_normal_df['item'] == i
            ]
            avg_label = item_records['label'].mean()

        new_labels[index] = avg_label

    # avg_labels.npy 저장
    np.save(path + 'avg_labels.npy', new_labels)

    print(f'{path}avg_labels.npy 생성 완료')

print('===== 전체 완료 =====')