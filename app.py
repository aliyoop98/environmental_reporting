import streamlit as st

st.subheader("Out-of-Range Events")
col_objs = st.columns(len(channels))

for i, ch in enumerate(channels):
    ch_lo, ch_hi = ranges[name].get(ch, (None, None))
    if ch_lo is None or ch not in sel.columns:
        continue

    df_ch = sel[['DateTime', ch]].copy()
    df_ch['OOR'] = df_ch[ch].apply(lambda v: pd.notna(v) and (v < ch_lo or v > ch_hi))
    df_ch['Group'] = (df_ch['OOR'] != df_ch['OOR'].shift(fill_value=False)).cumsum()

    events = []
    for gid, grp in df_ch.groupby('Group'):
        if not grp['OOR'].iloc[0]:
            continue
        start = grp['DateTime'].iloc[0]
        last_idx = grp.index[-1]
        if last_idx + 1 < len(sel) and not df_ch.loc[last_idx+1, 'OOR']:
            end = sel.loc[last_idx+1, 'DateTime']
        else:
            end = grp['DateTime'].iloc[-1]
        duration = max((end - start).total_seconds() / 60, 0)
        events.append({'Start': start, 'End': end, 'Duration(min)': duration})

    with col_objs[i]:
        st.markdown(f"### {ch} OOR Events")
        if events:
            ev_df = pd.DataFrame(events)
            total = ev_df['Duration(min)'].sum()
            incident = total >= 60 or any(ev_df['Duration(min)'] > 60)
            st.table(ev_df)
            st.write(f"**Total OOR minutes:** {total:.1f}  ")
            st.write(f"**Incident:** {'YES' if incident else 'No'}")
        else:
            st.info("No out-of-range events detected.")
