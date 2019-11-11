GET_COUPON = "SELECT DISTINCT c.id AS coupon_id, cc.name AS coupon_name, cc.description AS coupon_description, \
   sst.name AS coupon_searchtags, v.name AS vendor_name, v.description  AS vendor_des, vv.slug AS vendor_cate_name, \
   vv.description AS vendor_cate_des, ss.name AS vendor_searchtags, vv3.address FROM vendors_vendor v \
     LEFT JOIN vendors_vendor_categories vvc ON v.id = vvc.vendor_id \
     LEFT JOIN vendors_vendorcategory vv on vvc.vendorcategory_id = vv.id \
     LEFT JOIN vendors_vendortagsearch vv2 on v.id = vv2.vendor_id \
     LEFT JOIN settings_searchtags ss ON vv2.search_tag_id = ss.id \
     INNER JOIN coupons_coupon cc on v.id = cc.vendor_id \
     LEFT JOIN coupons_coupontags cct ON cc.id = cct.coupon_id \
     LEFT JOIN settings_searchtags sst ON cct.search_tag_id = sst.id \
     INNER JOIN coupons_cataloguecoupon c on cc.id = c.coupon_id \
    LEFT JOIN vendors_vendorlocation vv3 on v.id = vv3.vendor_id \
WHERE NOW() BETWEEN c.start AND c.end AND c.status='2' AND v.email NOT LIKE '%nguyen.do.bao.anh@tee-coin.com%' " \
                                      "AND v.parent_id IS NULL AND cc.id NOT IN (6, 7) ORDER BY coupon_id"

GET_VENDOR_RATING = "SELECT DISTINCT cc.vendor_id, cc.rating, cc.author_id AS user_id, gender, " \
                    "v.country_id AS vd_country_id FROM comments_comment cc INNER JOIN  \
                    vendors_vendor vv on vv.id = cc.vendor_id INNER JOIN accounts_account aa on cc.author_id = aa.id " \
                    "INNER JOIN vendors_vendorlocation v on vv.id = v.vendor_id WHERE vv.email NOT LIKE '%nguyen.do.bao.anh@tee-coin.com%'"

GET_VENDOR_FAVORITE = "SELECT ff.vendor_id, (CASE WHEN is_favorite THEN 5 ELSE 4 END) AS rating, ff.user_id, " \
                      "aa.gender, v.country_id AS vd_country_id FROM favorites_favoritevendor ff " \
                      "INNER JOIN vendors_vendor vv on ff.vendor_id = vv.id " \
                      "INNER JOIN accounts_account aa on ff.user_id = aa.id " \
                      "LEFT JOIN vendors_vendorlocation v on vv.id = v.vendor_id WHERE vv.email NOT LIKE '%nguyen.do.bao.anh@tee-coin.com%'"

GET_VENDOR_CONTENT = "SELECT DISTINCT v.id AS vendor_id, v.name AS vendor_name, v.description AS vendor_des, " \
                     "vv.slug AS vendor_cate_name, vv.description AS vendor_cate_des, ss.name AS vendor_searchtags, vv3.address " \
                     "FROM vendors_vendor v LEFT JOIN vendors_vendor_categories vvc ON v.id = vvc.vendor_id " \
                     "LEFT JOIN vendors_vendorcategory vv on vvc.vendorcategory_id = vv.id " \
                     "LEFT JOIN vendors_vendortagsearch vv2 on v.id = vv2.vendor_id " \
                     "LEFT JOIN settings_searchtags ss ON vv2.search_tag_id = ss.id " \
                     "LEFT JOIN vendors_vendorlocation vv3 on v.id = vv3.vendor_id " \
                     "INNER JOIN coupons_coupon cc ON v.id = cc.vendor_id " \
                     "INNER JOIN coupons_cataloguecoupon c ON cc.id = c.coupon_id " \
                     "WHERE NOW() BETWEEN c.start AND c.end AND c.status='2' AND v.email NOT LIKE '%nguyen.do.bao.anh@tee-coin.com%' " \
                     "AND v.parent_id IS NULL and cc.id NOT IN (6, 7) ORDER BY vendor_id"

GET_COUPON_CONTENT = "SELECT DISTINCT vd.id AS vendor_id, cc.name AS coupon_name," \
                     "cc.description AS coupon_description, sst.name AS coupon_searchtags FROM vendors_vendor vd " \
                     "LEFT JOIN coupons_coupon cc ON vd.id = cc.vendor_id " \
                     "LEFT JOIN coupons_coupontags cct ON cc.id = cct.coupon_id " \
                     "LEFT JOIN settings_searchtags sst ON cct.search_tag_id = sst.id  " \
                     "INNER JOIN coupons_cataloguecoupon c on cc.id = c.coupon_id " \
                     "WHERE NOW() BETWEEN c.start AND c.end AND c.status='2'AND vd.email NOT LIKE '%nguyen.do.bao.anh@tee-coin.com%' " \
                     "AND vd.parent_id IS NULL AND cc.id NOT IN (6, 7) ORDER BY vendor_id"


def get_gender(account_ids):

    gender_query = "SELECT id AS actual_account_id, gender FROM accounts_account WHERE id IN {}".format(account_ids)

    return gender_query


def get_vendor_loc_based_on_vendor_ids(vendor_ids):

    location_query = "SELECT vendor_id AS actual_item_id, country_id AS actual_item_country_id FROM vendors_vendorlocation WHERE vendor_id IN {}".format(vendor_ids)

    return location_query


def get_vendor_loc_based_on_cata_coupon_ids(cata_coupon_ids):

    location_query = "SELECT DISTINCT cc.id AS actual_item_id, country_id AS actual_item_country_id " \
                     "FROM coupons_cataloguecoupon cc INNER JOIN coupons_coupon c on cc.coupon_id = c.id " \
                     "INNER JOIN vendors_vendor vv on c.vendor_id = vv.id " \
                     "INNER JOIN vendors_vendorlocation v on vv.id = v.vendor_id WHERE cc.id IN {}".format(cata_coupon_ids)

    return location_query