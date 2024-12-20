# Code Citations

## License: unknown
https://github.com/bbrewington/data-tools/tree/1b5bfc18cf7fe2f74fa8ed56c55cd4b1eeb61c48/chatgpt-responsible-ai/chatgpt_conversation.md

```
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define
```

