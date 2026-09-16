import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# Settings
# =========================================================
ATTACK_RATIOS = [0.01, 0.03, 0.05]
THRESHOLD = 4

ATTACK_OUTPUT_DIR = Path('./attack_output')
PROCESS_OUTPUT_DIR = Path('./Data/Process/MovieLens')


def k_core_filtering(select_data, k=5):
    itr = 0

    while True:
        pre_shape = select_data.shape[0]

        user_info = select_data.groupby('user').agg(
            {'label': ['count', 'mean']}
        )
        selected_users = user_info[
            user_info[('label', 'count')] >= k
        ].index
        select_data = select_data[
            select_data['user'].isin(selected_users)
        ]

        item_info = select_data.groupby('item').agg(
            {'label': ['count', 'mean']}
        )
        selected_items = item_info[
            item_info[('label', 'count')] >= k
        ].index
        select_data = select_data[
            select_data['item'].isin(selected_items)
        ]

        aft_shape = select_data.shape[0]

        print(
            "itr:", itr,
            "pre-shape:", pre_shape,
            "aft_shape:", aft_shape
        )

        if pre_shape == aft_shape:
            break

        itr += 1

    return select_data


def encoding_f(pd_data, user_map, item_map, thre=4):
    pd_data = pd_data.copy()

    pd_data['user'] = pd_data['user'].map(user_map)
    pd_data['item'] = pd_data['item'].map(item_map)
    pd_data['label'] = pd_data['label'].apply(
        lambda x: 1 if x > thre else 0
    )

    return pd_data


def process_attack_ratio(attack):
    ratio_str = f'{attack:.2f}'

    load_path = ATTACK_OUTPUT_DIR / f'ratings_attacked_{ratio_str}.dat'
    fake_users_path = ATTACK_OUTPUT_DIR / f'fake_users_{ratio_str}.txt'
    save_path = PROCESS_OUTPUT_DIR / str(attack)

    print("\n" + "=" * 70)
    print(f"Processing attack ratio: {ratio_str}")
    print("=" * 70)

    if not load_path.is_file():
        raise FileNotFoundError(f'Attacked ratings file not found: {load_path}')

    if not fake_users_path.is_file():
        raise FileNotFoundError(f'Fake user file not found: {fake_users_path}')

    # =====================================================
    # 1. Load attacked MovieLens data
    # =====================================================
    data = pd.read_csv(
        load_path,
        delimiter='::',
        engine='python',
        header=None,
        usecols=[0, 1, 2]
    )

    data.columns = ['user', 'item', 'label']

    fake_users = pd.read_csv(
        fake_users_path,
        header=None,
        names=['user']
    )['user'].to_numpy()

    print("user num:", data['user'].nunique())
    print("item num:", data['item'].nunique())
    print("intersection num:", data.shape[0])
    print(
        "data sparse:",
        data.shape[0] / data['user'].nunique() / data['item'].nunique()
    )
    print("label type:", sorted(data['label'].unique()))

    # =====================================================
    # 2. 5-core filtering
    # =====================================================
    select_data = k_core_filtering(data, k=5)

    print("user num after 5-core:", select_data['user'].nunique())
    print("item num after 5-core:", select_data['item'].nunique())
    print("intersection num after 5-core:", select_data.shape[0])

    # =====================================================
    # 3. Separate normal data and injected attack data
    # =====================================================
    normal_data = select_data[
        ~select_data['user'].isin(fake_users)
    ].copy()

    attack_data = select_data[
        select_data['user'].isin(fake_users)
    ].copy()

    print("\n[Attack separation]")
    print("normal interactions:", normal_data.shape[0])
    print("attack interactions:", attack_data.shape[0])
    print("fake users after 5-core:", attack_data['user'].nunique())

    if attack_data.empty:
        raise ValueError(
            f'No injected attack data was found for ratio {ratio_str}. '
            'Check attacked ratings and fake user files.'
        )

    # =====================================================
    # 4. Split ONLY normal data into train / valid / test
    #    60% / 20% / 20%
    # =====================================================
    idx = np.arange(normal_data.shape[0])

    np.random.seed(1024)
    np.random.shuffle(idx)

    n1 = int(normal_data.shape[0] * 0.6)
    n2 = int(normal_data.shape[0] * 0.8)

    idx1 = idx[:n1]
    idx2 = idx[n1:n2]
    idx3 = idx[n2:]

    train_normal = normal_data.iloc[idx1].copy()
    valid = normal_data.iloc[idx2].copy()
    test = normal_data.iloc[idx3].copy()

    # inject_attack.py가 만든 공격 데이터 전체를 unlearning 대상으로 사용
    train_unlearn = attack_data.copy()

    # 실제 poisoning 학습 데이터 = 정상 train + 공격 데이터
    train = pd.concat(
        [train_normal, train_unlearn],
        ignore_index=True
    )

    print("\n[Train composition]")
    print("train_normal:", train_normal.shape[0])
    print("train_unlearn (attack):", train_unlearn.shape[0])
    print("train total:", train.shape[0])

    # =====================================================
    # 5. Create user/item ID maps from training data
    # =====================================================
    user_map = dict(
        zip(
            train['user'].unique(),
            np.arange(train['user'].nunique())
        )
    )

    item_map = dict(
        zip(
            train['item'].unique(),
            np.arange(train['item'].nunique())
        )
    )

    # =====================================================
    # 6. Keep only valid/test users and items seen in train
    # =====================================================
    train_user = train['user'].unique()
    train_item = train['item'].unique()

    valid_ = valid[
        valid['user'].isin(train_user)
        & valid['item'].isin(train_item)
    ].copy()

    test_ = test[
        test['user'].isin(train_user)
        & test['item'].isin(train_item)
    ].copy()

    # =====================================================
    # 7. Re-index IDs and binarize ratings
    #
    # MovieLens:
    # rating > 4  -> 1
    # rating <= 4 -> 0
    # =====================================================
    train_normal_save = encoding_f(
        train_normal,
        user_map,
        item_map,
        thre=THRESHOLD
    )

    train_unlearn_save = encoding_f(
        train_unlearn,
        user_map,
        item_map,
        thre=THRESHOLD
    )

    train_save = encoding_f(
        train,
        user_map,
        item_map,
        thre=THRESHOLD
    )

    valid_save = encoding_f(
        valid_,
        user_map,
        item_map,
        thre=THRESHOLD
    )

    test_save = encoding_f(
        test_,
        user_map,
        item_map,
        thre=THRESHOLD
    )

    # 기존 label flip 코드는 사용하지 않음.
    # train_unlearn 자체가 inject_attack.py가 만든 실제 공격 데이터임.

    # =====================================================
    # 8. Save CSV files
    # =====================================================
    save_path.mkdir(parents=True, exist_ok=True)

    train_save.to_csv(
        save_path / 'train.csv',
        index=False
    )

    train_normal_save.to_csv(
        save_path / 'train_normal.csv',
        index=False
    )

    # 기존 후속 코드와의 호환성을 위해 파일명은 유지
    # 실제 내용은 랜덤 1%가 아니라 injected attack data
    train_unlearn_save.to_csv(
        save_path / 'train_random.csv',
        index=False
    )

    valid_save.to_csv(
        save_path / 'valid.csv',
        index=False
    )

    test_save.to_csv(
        save_path / 'test.csv',
        index=False
    )

    print("\nSaved to:", save_path)
    print(" - train.csv")
    print(" - train_normal.csv")
    print(" - train_random.csv")
    print(" - valid.csv")
    print(" - test.csv")


if __name__ == '__main__':
    for attack in ATTACK_RATIOS:
        process_attack_ratio(attack)

    print("\nAll attack ratios finished.")
