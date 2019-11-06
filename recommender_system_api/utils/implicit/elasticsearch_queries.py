def get_active_vendors():
    query = {
      "size": 10000,
      "_source": ["id", "vendor.id"],
      "query": {
        "bool": {
          "must": [
            {"match": {
              "status": "2"
            }},

            {
              "range": {
                "start": {
                  "lte": "now/s"

                }
              }
            },
            {
              "range": {
                "end": {
                  "gte": "now/s"
                }
              }
            }
          ]
        }
      },
      "aggs": {
        "unique_ids": {
          "terms": {
            "field": "vendor.id"
          }
        }
      }
    }
    return query


def get_vendor_detail_views(path):
    query = {
      "size": 10000,
      "_source": ["account", "path"],
      "query": {
        "bool": {
          "must": [
            {"match": {
              "path": "/detail/"
            }}
          ],
          "must_not": [
            {"match": {
              "path": "place_details"
            }}
          ],
          "filter": {
            "terms": {
              "path": path
            }
          }
        }
      },
      "aggs": {
        "distinct_records": {
          "terms": {
            "field": "path.keyword",
            "script": "[doc['account'].value, doc['path'].value]"
          }
        }
      },
      "sort": [
        {
          "timestamp": {
            "order": "desc"
          }
        }
      ]
    }
    return query


def get_coupon_detail_views(path):
    query = {
      "_source": ["account", "path"],
      "size": 100,
      "query": {
        "bool": {
          "must": [
            {
              "query_string": {
                "fields": ["path"],
                "query": "catalogues/[0-9]{1,5}/"
              }
            }
          ],
          "filter": {
            "terms": {
              "path": path
            }
          }
        }
      },
      "aggs": {
        "distinct_records": {
          "terms": {
            "field": "path.keyword",
            "script": "[doc['account'].value, doc['path'].value]"
          }
        }
      },
      "sort": [
        {
          "timestamp": {
            "order": "desc"
          }
        }
      ]
    }

    return query
