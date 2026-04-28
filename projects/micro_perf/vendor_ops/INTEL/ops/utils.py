import random


# generator for prefill mode
def generate_prefill_data(
    q_seq_len, cache_len
):
    q_lens = [q_seq_len]
    accum_q_lens = [0, q_seq_len]
    cache_lens = [cache_len]
    cache_slot_ids = [0]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


# generator for prefill_session_cache mode
def generate_prefill_session_cache_data(
        batch_size,
        target_q_len,
        aver_cache_len
):
    # random q_len, accum to target_q_len
    aver_q_len = target_q_len // batch_size
    q_len_remainder = target_q_len % batch_size
    q_len_offset = aver_q_len // 10
    q_lens = []
    for i in range(batch_size):
        q_lens.append(aver_q_len + (1 if i < q_len_remainder else 0))
    for i in range(batch_size):
        q_lens[i] += random.randint(-q_len_offset, q_len_offset)

    # accum q_lens
    accum_q_lens = [0]
    for i in range(batch_size):
        accum_q_lens.append(accum_q_lens[-1] + q_lens[i])

    # random cache_lens
    cache_lens = [aver_cache_len for _ in range(batch_size)]
    cache_offset = aver_cache_len // 10
    for i in range(batch_size):
        cache_lens[i] += random.randint(-cache_offset, cache_offset)

    # sequential cache_slot_ids
    cache_slot_ids = [i for i in range(batch_size)]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


# generator for decode mode
def generate_decode_data(
    batch_size,
    q_seq_len,
    aver_cache_len
):
    # fixed q_len
    q_lens = [q_seq_len for _ in range(batch_size)]

    # accum q_lens
    accum_q_lens = [0]
    for i in range(batch_size):
        accum_q_lens.append(accum_q_lens[-1] + q_lens[i])

    # random cache_lens
    cache_lens = [aver_cache_len for _ in range(batch_size)]
    cache_offset = aver_cache_len // 10
    for i in range(batch_size):
        cache_lens[i] += random.randint(-cache_offset, cache_offset)

    # sequential cache_slot_ids
    cache_slot_ids = [i for i in range(batch_size)]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens
