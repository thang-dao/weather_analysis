import pandas as pd


def group_by_province(provinces, group_name):
    flag = False
    for prov in provinces:
        small_df = df[df.Province == prov]
        print(prov, small_df.shape)
        small_df = small_df.drop(['Province'], axis=1)
        if not flag:
            group_df = small_df
            flag = True
        else:
            group_df = pd.concat((group_df, small_df), axis=0)
    group_df.to_csv('data/{}.csv'.format(group_name))
    

if __name__=='__main__':
    df = pd.read_csv ('data/weather_data_VietNam_[2021].csv')
    # group_by_province(['Hà Nội', 'Bắc Ninh', 'Hà Nam', 'Hải Dương', 
    #                    'Hải Phòng', 'Hưng Yên', 'Nam Định', 'Thái Bình'], group_name='Bắc Bộ')
    # group_by_province(['Đà Nẵng', 'Quảng Nam', 'Quảng Ngãi', 'Bình Định',
    #                    'Phú Yên', 'Khánh Hòa', 'Ninh Thuận', 'Bình Thuận'], group_name='Trung Bộ')
    # group_by_province(['Thành phố Hồ Chí Minh', 'Bà Rịa - Vũng Tàu', 'Bình Dương', 
    #                    'Bình Phước', 'Đồng Nai', 'Tây Ninh', 'An Giang', 'Bạc Liêu'], group_name='Nam Bộ')
    group_by_province(['Tây Ninh'], group_name='Tây Ninh')