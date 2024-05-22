import polars as pl

def unfold_tree(base_df: pl.DataFrame, base_col: str, parent_col: str | None = None,
                forlimit: int = 16) -> pl.DataFrame:
    """Widens relational dataset so that top-level tree/parent is first column, and children are subsequent columns
    
    Assumes long relational dataset with columns named: `base_col`, f'{base_col}_name', and (if parent_col is none) f'{base_col}_rollup'. Assumes lowest-level child value never appears as a rollup value. Function takes distinct combination of base_col, base_col_name, and parent_col to create dataset.
    
    params:
        base_df (pl.DataFrame): Polars Dataframe with relationship information. 

        base_col (str): String column representing name of the entity, and prefix of the name of the entity

        parent_col (str, None): String of the parent/rollup column name. Default None, which sets name to f'{base_col}_rollup'

        forlimit (int): Max number of times to loop through the dataset looking for additional parents. Defualt 16.

    returns:
        pl.DataFrame with each row a unique child/leaf, with columns for each level of parents/branches

    """

    if parent_col is None:
        parent_col = f'{base_col}_rollup'

    base_df = base_df.select(
        pl.col(base_col),
        pl.col(f'{base_col}_name'),
        pl.col(parent_col)
    ).unique()

    trees = base_df.filter(
        pl.col(parent_col).is_in(
            base_df.select(
                pl.col(base_col).unique()
            ).to_series()
        ) == False
    )

    trees = trees.select(
        pl.col(parent_col).alias(f'{base_col}_tree_1'),
    )

    for n in range(1, forlimit):

        # print(n)

        trees = trees.join(
            base_df,
            right_on=parent_col,
            left_on=f'{base_col}_tree_{n}',
            how='left'
        ).select(
            *[col for col in trees.columns],
            pl.col(f'{base_col}_name').alias(f'{base_col}_tree_{n + 1}_name'),
            pl.col(base_col).alias(f'{base_col}_tree_{n + 1}')
        )

        trees_test = all(
            trees.select(
                pl.col(f'{base_col}_tree_{n + 1}').is_null()
            ).to_series()
        )

        if trees_test:
            trees = trees.drop(f'{base_col}_tree_{n + 1}')

            maxlimit = n

            break

    # identify child entity
    trees = trees.with_columns(
        pl.coalesce(
            *[pl.col(f'{base_col}_tree_{n}' for n in range(maxlimit, 1, -1))]
        ).alias(f'{base_col}_child'),
        pl.coalesce(
            *[pl.col(f'{base_col}_tree_{n}_name' for n in range(maxlimit, 1, -1))]
        ).alias(f'{base_col}_child_name'),
        pl.concat_str(
            *[pl.col(f'{base_col}_tree_{n}').fill_null('') for n in range(1, maxlimit + 1, 1)],
            separator=', '
        ).str.replace_all(' ,', '').str.replace(', $', '').alias(f'{base_col}_tree_desc')
    )

    return trees.unique()

