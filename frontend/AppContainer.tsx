import React from 'react';
import { Notifications } from 'expo';
import { Text, View } from 'react-native';

import registerForPushNotificationsAsync from './js/notifications';
import { data, getData } from './js/display';

export default class AppContainer extends React.Component {
    state = {
        notification: {},
    };

    componentDidMount() {
        registerForPushNotificationsAsync();

        setInterval(() => {getData()}, 1000);

        // Handle notifications that are received or selected while the app
        // is open. If the app was closed and then opened by tapping the
        // notification (rather than just tapping the app icon to open it),
        // this function will fire on the next tick after the app starts
        // with the notification data.
        this._notificationSubscription = Notifications.addListener(this._handleNotification);
    }

    _handleNotification = notification => {
        // do whatever you want to do with the notification
        this.setState({ notification: notification });
    };

    render() {
        return (
            <View style={{ flex: 1, justifyContent: 'center' }}>
                {data.map((row, index) => (
                    <Text key={index}>Row: {row}</Text>
                ))}
            </View>
        );
    }
}